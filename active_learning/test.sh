#!/bin/bash

# # Install required packages
# pip install blobfile tqdm

# # Define the log directory
LOGDIR=./logs/dry_run

# # Execute the training script with specified parameters

# ------------------------------------------------------------------------------
# NOTE: This script trains on 1 GPU
# ------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 

python3 -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  active_learning/example.py \
  --model                   wrn \
  --depth                   28 \
  --width                   12 \
  --dropout_rate            0.3 \
  --norm                    batch \
  --budget_ratio            0.1 \
  --sampling_method         uncertainty \
  --num_epochs              10 \
  --lr                      1e-4 \
  --decay_epochs            60 120 160 \
  --decay_rate              0.2 \
  --data_dir                ./data \
  --dataset                 bloodmnist_28 \
  --batch_size              128 \
  --in_channels             3 \
  --num_classes             8 \
  --image_size              28 \
  --eval_freq               10 \
  --log_dir                 $LOGDIR \

