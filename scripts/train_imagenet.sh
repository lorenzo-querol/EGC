#!/bin/bash

# Define the log directory
export OPENAI_LOGDIR=./openai_log/imagenet

# Execute the training script with specified parameters
python3 -m torch.distributed.run \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  scripts/image_train_ldm_clsonline_egc.py \
  --attention_resolutions   32,16,8 \
  --class_cond              True \
  --num_classes             1000 \
  --diffusion_steps         1000 \
  --dropout                 0.0 \
  --image_size              32 \
  --learn_sigma             False \
  --noise_schedule          linear \
  --linear_start            0.00085 \
  --linear_end              0.012 \
  --num_channels            384 \
  --num_head_channels       64 \
  --num_res_blocks          2 \
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
  --val_data_dir            ./data/imagenet256_features_val \
  --data_dir                ./data/imagenet256_features \
  --cls_data_dir            ./data/imagenet/train \
  --ce_weight               0.005 \
  --eval_interval           5000 \
  --save_interval           10000 \
  --label_smooth            0.2 \
  --grad_clip               1.0 \
  --channel_mult            1,2,4 \
  --microbatch              8 \
  --in_channels             4 \
  --sample_z                False \
  --autoencoder_path        ./autoencoder_kl.pth \
  --use_spatial_transformer True \
  --context_dim             512 \
  --transformer_depth       1