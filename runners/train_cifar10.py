"""
Train a diffusion model on images.
"""

import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to sys.path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import get_val_data, load_ldm_data, get_val_ldm_data, load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    egc_model_and_diffusion_defaults,
    create_egc_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


import torch.distributed as dist
import multiprocessing
import blobfile as bf
import wandb


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.egc_model_and_diffusion_defaul
    return logger.get_dir()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    if dist.get_rank() == 0:
        cfg = args_to_dict(args, egc_model_and_diffusion_defaults().keys())
        name = logger.get_dir().split("/")[-1]
        wandb.init(project=f"EGC", config=cfg, name=name)

    model, diffusion = create_egc_model_and_diffusion(**args_to_dict(args, egc_model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loaders...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.input_size,
        class_cond=args.class_cond,
        random_flip=True,
        random_crop=True,
        deterministic=True,
        in_channels=args.in_channels,
    )

    if args.cls_batch_size > 0:
        data_cls = load_data(
            data_dir=args.cls_data_dir,
            batch_size=args.cls_batch_size,
            image_size=args.input_size,
            class_cond=args.class_cond,
            random_flip=True,
            random_crop=True,
            deterministic=True,
            in_channels=args.in_channels,
        )
    else:
        data_cls = None

    if args.val_data_dir:
        val_data = get_val_data(
            data_dir=args.val_data_dir,
            batch_size=args.val_batch_size,
            image_size=args.input_size,
            class_cond=args.class_cond,
            random_flip=False,
            in_channels=args.in_channels,
        )
    else:
        val_data = None

    logger.log("Training...")
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


def create_argparser():
    defaults = dict(
        data_dir="",
        cls_data_dir="",
        val_data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        cls_batch_size=0,
        val_batch_size=0,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        local_rank=0,
        ce_weight=0.0,
        eval_interval=5000,
        label_smooth=0.0,
        use_hdfs=False,
        grad_clip=1.0,
        prepare_data=False,
        img_num=1_281_167 * 2,
        autoencoder_path=None,
        betas1=0.9,
        betas2=0.999,
        cls_cond_training=False,
        train_classifier=True,
        sample_z=False,
        double_z=True,
        scale_factor=0.18215,
        autoencoder_stride="8",
        autoencoder_type="KL",
        warm_up_iters=-1,
        input_size=32,
    )
    defaults.update(egc_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
