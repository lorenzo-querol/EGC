from contextlib import contextmanager
import json
import logging
from argparse import Namespace
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from active_learning.data import ImageAugDataset, ImageDataset
from active_learning.sampling import random_sampling, uncertainty_sampling
from guided_diffusion import dist_util
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    args_to_dict,
    create_egc_model_and_diffusion,
    egc_model_and_diffusion_defaults,
)
from guided_diffusion.train_util import TrainLoop
from models.wide_resnet import Wide_ResNet
from runners.wrn_runner import WRNTrainer


class SamplingMethod(Enum):
    """Supported sampling methods for active learning."""

    RANDOM = "random"
    UNCERTAINTY = "uncertainty"

    @classmethod
    def from_string(cls, value: str) -> "SamplingMethod":
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unsupported sampling method: {value}. " f"Supported methods: {[m.value for m in cls]}")


@contextmanager
def distributed_context():
    """Context manager for handling distributed training setup and cleanup."""
    try:
        dist_util.setup_dist()
        yield
    except Exception as e:
        logging.error(f"Error during distributed setup: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@dataclass
class ModelManager:
    args: Namespace
    device: torch.device

    def create_egc_model(self):
        model, diffusion = create_egc_model_and_diffusion(**args_to_dict(self.args, egc_model_and_diffusion_defaults().keys()))
        model.to(self.device)
        schedule_sampler = create_named_schedule_sampler(self.args.schedule_sampler, diffusion)
        return model, diffusion, schedule_sampler

    def create_wrn_model(self):
        model = Wide_ResNet(self.args.depth, self.args.width, self.args.in_channels, self.args.num_classes, self.args.dropout_rate, self.args.norm)
        model.to(self.device)
        optim = torch.optim.SGD(model.parameters(), self.args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.args.decay_epochs, gamma=self.args.decay_rate)

        return model, optim, scheduler


@dataclass
class ALState:
    """Dataclass to hold the active learning state"""

    iteration: int
    l_set: List[str]
    u_set: List[str]
    checkpoint_path: Optional[str]
    sampling_method: str
    budget_ratio: float
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ALState":
        return cls(**data)


class ActiveLearner:
    def __init__(self, budget_ratio: float, sampling_method: str, args: Namespace, resume_from: Optional[str] = None):
        """
        Initialize the ActiveLearner with the given configuration.

        Args:
            budget_ratio: Ratio of labeled samples to total samples
            sampling_method: Sampling method to use for querying new samples
            args: Command-line arguments
            resume_from: Path to resume from a previous state
        """
        self.args = args
        self.sampling_method = SamplingMethod.from_string(sampling_method)
        self.budget_ratio = budget_ratio

        # Setup paths
        self.base_dir = Path(args.log_dir)
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.states_dir = self.base_dir / "states"

        # Create necessary directories
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)

        # Initialize or restore state
        if resume_from:
            self._restore_state(resume_from)
        else:
            self._initialize_new_state()

    def _initialize_new_state(self) -> None:
        """Initialize a new active learning state"""
        self.train_files = self._list_image_files(self.base_dir / "train")
        self.val_files = self._list_image_files(self.base_dir / "val")

        self.total_size = len(self.train_files)
        self.budget_size = int(self.total_size * self.budget_ratio)

        self.current_iteration = 0
        self.l_set = self.train_files[: self.budget_size]
        self.u_set = self.train_files[self.budget_size :]

        # Save initial state
        self._save_current_state()

    def _restore_state(self, state_path: str) -> None:
        """
        Restore active learning state from a saved state file.

        Args:
            state_path: Path to the state file to restore from
        """
        state_path = Path(state_path)
        if not state_path.exists():
            raise ValueError(f"State file not found: {state_path}")

        # Load state
        with open(state_path, "r") as f:
            state_dict = json.load(f)
            state = ALState.from_dict(state_dict)

        # Restore state
        self.current_iteration = state.iteration
        self.l_set = state.l_set
        self.u_set = state.u_set
        self.total_size = state.total_samples
        self.budget_size = int(self.total_size * self.budget_ratio)

        # Validate restored state
        if len(self.l_set) + len(self.u_set) != self.total_size:
            raise ValueError("Corrupted state: total samples mismatch")

        logging.info(f"Restored state from iteration {self.current_iteration}")
        logging.info(f"Labeled samples: {len(self.l_set)}, Unlabeled samples: {len(self.u_set)}")

    def _save_current_state(self) -> Path:
        """
        Save current active learning state.

        Returns:
            Path to the saved state file
        """
        state = ALState(
            iteration=self.current_iteration,
            l_set=self.l_set,
            u_set=self.u_set,
            checkpoint_path=str(self._get_latest_checkpoint_path()),
            sampling_method=self.sampling_method.value,
            budget_ratio=self.budget_ratio,
            total_samples=self.total_size,
        )

        # Save state to file
        state_path = self.states_dir / f"state_iter_{self.current_iteration}.json"
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        return state_path

    def _get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get the path to the latest model checkpoint for current iteration"""
        checkpoint_pattern = f"model_iter_{self.current_iteration}_*.pt"
        checkpoints = list(self.checkpoints_dir.glob(checkpoint_pattern))
        return max(checkpoints, default=None)

    def _save_checkpoint(self, model: Module, optimizer: torch.optim.Optimizer) -> Path:
        """
        Save a model checkpoint.

        Args:
            model: The model to save
            optimizer: The optimizer to save

        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = self.checkpoints_dir / f"model_iter_{self.current_iteration}_step_{self.current_step}.pt"

        checkpoint = {
            "iteration": self.current_iteration,
            "epoch": self.current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": self.args,
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint(self, model: Module, optimizer: torch.optim.Optimizer) -> None:
        """
        Load the latest checkpoint for the current iteration.

        Args:
            model: The model to load state into
            optimizer: The optimizer to load state into
        """
        checkpoint_path = self._get_latest_checkpoint_path()
        if checkpoint_path is None:
            logging.warning(f"No checkpoint found for iteration {self.current_iteration}")
            return

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_step = checkpoint["step"]

        logging.info(f"Restored checkpoint from iteration {self.current_iteration}, step {self.current_step}")

    def _list_image_files(self, path: Path) -> List[str]:
        """Recursively list all image files in the given directory."""

        valid_extensions = {".jpg", ".jpeg", ".png", ".gif"}
        return [str(f) for f in path.rglob("*") if f.suffix.lower() in valid_extensions]

    def _create_dataset(self, image_paths: List[str], class_cond: bool = False, augment: bool = False) -> Dataset:
        """Create a dataset from the given image paths."""

        dataset_cls = ImageAugDataset if augment else ImageDataset
        classes = None

        if class_cond:
            class_names = [Path(path).stem.split("_")[0] for path in image_paths]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        return dataset_cls(
            resolution=self.args.image_size,
            image_paths=image_paths,
            in_channels=self.args.in_channels,
            classes=classes,
            shard=dist.get_rank(),
            num_shards=dist.get_world_size(),
        )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader from the given dataset."""

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=8,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=True,
        )

    def get_train_dataloaders(self, class_cond: bool = False, deterministic: bool = False) -> Tuple[DataLoader, DataLoader]:
        """
        Get training dataloaders for both full and labeled datasets.

        Args:
            class_cond: Whether to include class information
            deterministic: Whether to use deterministic ordering

        Returns:
            Tuple of (full training dataloader, labeled set dataloader)
        """

        full_dataset = self._create_dataset(self.l_set + self.u_set, class_cond=class_cond, augment=True)
        labeled_dataset = self._create_dataset(self.l_set, class_cond=class_cond, augment=True)

        shuffle = not deterministic
        return (self._create_dataloader(full_dataset, shuffle=shuffle), self._create_dataloader(labeled_dataset, shuffle=shuffle))

    def get_val_dataloader(self, class_cond: bool = False) -> DataLoader:
        """Get validation dataloader."""

        dataset = self._create_dataset(self.val_files, class_cond=class_cond)
        return self._create_dataloader(dataset, shuffle=False)

    def get_unlabeled_dataloader(self) -> DataLoader:
        """Get dataloader for unlabeled samples."""

        dataset = self._create_dataset(self.u_set)
        return self._create_dataloader(dataset, shuffle=False)

    def query_samples(self, model: Module) -> None:
        """
        Query new samples using the specified sampling method.

        Args:
            model: Model to use for uncertainty sampling
        """

        if self.budget_size > len(self.u_set):
            self.budget_size = len(self.u_set)

        u_array = np.array(self.u_set)

        if self.sampling_method == SamplingMethod.RANDOM:
            l_selected, u_remaining = random_sampling(u_array, self.budget_size)

        elif self.sampling_method == SamplingMethod.UNCERTAINTY:
            u_loader = self.get_unlabeled_dataloader()
            l_selected, u_remaining = uncertainty_sampling(model, u_loader, u_array, self.budget_size)

        self.l_set.extend(l_selected.tolist())
        self.u_set = u_remaining.tolist()

        self._save_sets()

    def learn(self, args) -> None:
        """Train the model in an active learning loop."""

        with distributed_context():
            device = dist_util.dev()

            if args.model == "EGC":
                self._run_egc_training(args, device)
            elif args.model == "wrn":
                self._run_wrn_training(args, device)
            else:
                raise ValueError(f"Unsupported model type: {args.model}")

    def _run_egc_training(self, args, device) -> None:
        """Run EGC model training loop."""

        model, diffusion, schedule_sampler = ModelManager.create_egc_model(args, device)

        while True:
            data, data_cls = self.get_train_dataloaders(class_cond=args.class_cond, deterministic=True)
            val_data = self.get_val_dataloader(class_cond=args.class_cond)

            TrainLoop(
                model=model,
                diffusion=diffusion,
                data=data,
                data_cls=data_cls,
                val_data=val_data,
                batch_size=args.batch_size,
                cls_batch_size=args.cls_batch_size,
                microbatch=args.microbatch,
                lr=args.lr,
                ema_rate=args.ema_rate,
                log_interval=args.log_interval,
                save_interval=args.save_interval,
                resume_checkpoint=args.resume_checkpoint,
                use_fp16=args.use_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
                schedule_sampler=schedule_sampler,
                weight_decay=args.weight_decay,
                lr_anneal_steps=args.lr_anneal_steps,
                ce_weight=args.ce_weight,
                eval_interval=args.eval_interval,
                label_smooth=args.label_smooth,
                use_hdfs=args.use_hdfs,
                grad_clip=args.grad_clip,
                local_rank=args.local_rank,
                autoencoder_path=args.autoencoder_path,
                betas=(args.betas1, args.betas2),
                cls_cond_training=args.cls_cond_training,
                train_classifier=args.train_classifier,
                scale_factor=args.scale_factor,
                autoencoder_stride=args.autoencoder_stride,
                autoencoder_type=args.autoencoder_type,
                warm_up_iters=args.warm_up_iters,
                encode_cls=False,
            ).run_loop()

            if self._should_stop_training():
                break

            self.query_samples(model)

    def _run_wrn_training(self, args, device) -> None:
        """Run Wide-ResNet training loop."""

        model, optim, scheduler = ModelManager(args, device).create_wrn_model()

        while True:
            _, data_cls = self.get_train_dataloaders(class_cond=True, deterministic=True)
            val_data = self.get_val_dataloader(class_cond=True)

            trainer = WRNTrainer(
                model=model,
                optim=optim,
                scheduler=scheduler,
                train_data=data_cls,
                val_data=val_data,
            )

            # Load checkpoint if exists
            self._load_checkpoint(model, optim)

            # Run training loop with periodic checkpointing
            def save_callback(epoch):
                self.current_epoch = epoch
                self._save_checkpoint(model, optim)

            trainer.run_loop(num_epochs=args.num_epochs, eval_freq=args.eval_freq, save_callback=save_callback)

            if self._should_stop_training():
                break

            self.current_iteration += 1
            self.query_samples(model)
            self._save_current_state()

    def _should_stop_training(self) -> bool:
        """Check if training should stop."""

        if len(self.u_set) == 0:
            return True

        return False
