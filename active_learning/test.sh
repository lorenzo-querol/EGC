#!/bin/bash

# # Install required packages
# pip install blobfile tqdm

# # Define the log directory
# OPENAI_LOGDIR=./logs/mnist

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
  --budget_ratio            0.1 \
  --num_classes             10 \
  --dropout_rate            0.3 \
  --image_size              28 \
  --lr                      1e-4 \
  --batch_size              128 \
  --data_dir                .data/bloodmnist_28 \

