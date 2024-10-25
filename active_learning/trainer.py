from contextlib import contextmanager
import logging
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from active_learning.data import ImageAugDataset, ImageDataset
from active_learning.sampling import random_sampling, uncertainty_sampling
from active_learning.utils import ActiveLearnerConfig, DatasetConfig, TrainConfig
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
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class ModelManager:
    """Handles model initialization and training for different model types."""

    @staticmethod
    def create_egc_model(args, device):
        model, diffusion = create_egc_model_and_diffusion(**args_to_dict(args, egc_model_and_diffusion_defaults().keys()))
        model.to(device)
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        return model, diffusion, schedule_sampler

    @staticmethod
    def create_wrn_model(args, device):
        model = Wide_ResNet(args.depth, args.width, args.in_channels, args.num_classes, args.dropout_rate, args.norm)
        model.to(device)
        optim = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        return model, optim


class ActiveLearner:
    def __init__(self, budget_ratio: float, sampling_method: str, args: Namespace):
        """
        Initialize the ActiveLearner with the given configuration.

        Args:
            budget_ratio: Ratio of labeled samples to total samples
            sampling_method: Sampling method to use for querying new samples
            args: Command-line arguments
        """
        self.args = args
        self.sampling_method = SamplingMethod.from_string(sampling_method)

        # Initialize dataset splits
        self.train_files = self._list_image_files(args.data_dir / "train")
        self.val_files = self._list_image_files(args.data_dir / "val")

        # Initialize labeled and unlabeled sets
        self.total_size = len(self.train_files)
        self.budget_size = int(self.total_size * budget_ratio)

        # Split the dataset into labeled and unlabeled sets
        self.l_set = self.train_files[: self.budget_size]
        self.u_set = self.train_files[self.budget_size :]

        # Save the initial sets to disk for resuming training
        self._save_sets()

    def _list_image_files(self, path: Path) -> List[str]:
        """Recursively list all image files in the given directory."""

        valid_extensions = {".jpg", ".jpeg", ".png", ".gif"}
        return [str(f) for f in path.rglob("*") if f.suffix.lower() in valid_extensions]

    def _save_sets(self) -> None:
        """Save current labeled and unlabeled sets to disk."""

        for name, dataset in [("l_set", self.l_set), ("u_set", self.u_set)]:
            save_path = self.config.dataset_path / f"{name}.txt"
            save_path.write_text("\n".join(dataset))

    def _save_class_distribution(self) -> pd.DataFrame:
        """Analyze and return the class distribution in the labeled set."""

        class_names = [Path(path).stem.split("_")[0] for path in self.l_set]
        class_counts = pd.Series(class_names).value_counts()

        data = {"class": class_counts.index, "count": class_counts.values, "percentage": (class_counts.values / len(self.l_set) * 100).round(2)}
        df = pd.DataFrame(data)
        df.to_csv(self.config.dataset_path / "class_distribution.csv", index=False)

    def _create_dataset(self, image_paths: List[str], class_cond: bool = False, augment: bool = False) -> Dataset:
        """Create a dataset from the given image paths."""

        dataset_cls = ImageAugDataset if augment else ImageDataset
        classes = None

        if class_cond:
            class_names = [Path(path).stem.split("_")[0] for path in image_paths]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]

        return dataset_cls(
            resolution=self.config.image_size,
            image_paths=image_paths,
            in_channels=self.config.in_channels,
            classes=classes,
            shard=dist.get_rank(),
            num_shards=dist.get_world_size(),
        )

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader from the given dataset."""

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
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
        """Train the model in an active learning loop with proper distributed cleanup."""

        with distributed_context():
            device = dist_util.dev()

            if args.model == "EGC":
                self._run_egc_training(device, args)
            elif args.model == "wrn":
                self._run_wrn_training(device)
            else:
                raise ValueError(f"Unsupported model type: {args.model}")

    def _run_egc_training(self, device, args):
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

    def _run_wrn_training(self, device):
        """Run Wide-ResNet training loop."""

        model, optim = ModelManager.create_wrn_model(device)

        while True:
            _, data_cls = self.get_train_dataloaders(class_cond=True, deterministic=True)
            val_data = self.get_val_dataloader(class_cond=True)

            WRNTrainer(
                model=model,
                optim=optim,
                train_data=data_cls,
                val_data=val_data,
                device=device,
            ).run_loop(
                num_epochs=100,
                eval_every_epoch=10,
            )

            if self._should_stop_training():
                break

            self.query_samples(model)

    def _should_stop_training(self) -> bool:
        """Check if training should stop."""

        if len(self.u_set) == 0:
            return True

        return False
