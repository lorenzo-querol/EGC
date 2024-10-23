#!/bin/bash

# Install required packages
pip install blobfile tqdm

# Define the log directory
OPENAI_LOGDIR=./logs/mnist

# Execute the training script with specified parameters

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
  runners/train_cifar10.py \
  --attention_resolutions   16,8 \
  --class_cond              True \
  --num_classes             10 \
  --diffusion_steps         1000 \
  --dropout                 0.0 \
  --image_size              28 \
  --learn_sigma             False \
  --noise_schedule          cosine \
  --linear_start            0.00085 \
  --linear_end              0.012 \
  --num_channels            192 \
  --num_head_channels       64 \
  --num_res_blocks          3 \
  --resblock_updown         False \
  --use_new_attention_order True \
  --use_fp16                False \
  --use_scale_shift_norm    False \
  --pool                    sattn \
  --lr                      1e-4 \
  --weight_decay            0.0 \
  --batch_size              128 \
  --val_batch_size          128 \
  --cls_batch_size          128 \
  --val_data_dir            ./data/mnist/val \
  --data_dir                ./data/mnist/train \
  --cls_data_dir            ./data/mnist/train \
  --ce_weight               0.001 \
  --eval_interval           5000 \
  --save_interval           5000 \
  --label_smooth            0.2 \
  --grad_clip               1.0 \
  --channel_mult            1,2,2 \
  --microbatch              8 \
  --in_channels             1 \
  --sample_z                False \
  --use_spatial_transformer True \
  --context_dim             512 \
  --transformer_depth       1