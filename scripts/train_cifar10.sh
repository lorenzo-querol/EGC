#!/bin/bash

# Define the log directory
export OPENAI_LOGDIR=./openai_log/cifar10

# Execute the training script with specified parameters
python3 -m torch.distributed.run \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  runners/train_egc.py \
  --attention_resolutions   16,8 \
  --class_cond              True \
  --num_classes             10 \
  --diffusion_steps         1000 \
  --dropout                 0.0 \
  --image_size              32 \
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
  --batch_size              64 \
  --cls_batch_size          64 \
  --val_batch_size          128 \
  --val_data_dir            ./data/cifar10/test \
  --data_dir                ./data/cifar10/train \
  --cls_data_dir            ./data/cifar10/train \
  --ce_weight               0.001 \
  --eval_interval           5000 \
  --save_interval           10000 \
  --label_smooth            0.2 \
  --grad_clip               1.0 \
  --channel_mult            1,2,2 \
  --microbatch              8 \
  --in_channels             3 \
  --sample_z                False \
  --use_spatial_transformer True \
  --context_dim             512 \
  --transformer_depth       1