#!/usr/bin/env bash

# Install required package
pip install blobfile

# Parse arguments
LOGDIR=$1
GPU=$2
NODE=$3
RANK=$4
MADDR=$5
CKPT_PATH=$6
OPT=${@:7}

# Set the log directory
export OPENAI_LOGDIR=./openai_log/${LOGDIR}

# Execute the distributed training script
python3 -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 
  scripts/image_sample_cond_ldm_egc.py \
  --num_samples 50000 \
  --autoencoder_path ./autoencoder_kl.pth \
  --attention_resolutions 32,16,8 \
  --class_cond True \
  --diffusion_steps 1000 \
  --dropout 0.0 \
  --image_size 32 \
  --learn_sigma False \
  --noise_schedule linear \
  --linear_start 0.00085 \
  --linear_end 0.012 \
  --num_channels 384 \
  --num_head_channels 64 \
  --num_res_blocks 2 \
  --resblock_updown False \
  --use_new_attention_order True \
  --use_fp16 False \
  --use_scale_shift_norm False \
  --pool sattn \
  --batch_size 8 \
  --channel_mult 1,2,4 \
  --in_channels 4 \
  --classifier_scale 6.0 \
  --use_spatial_transformer True \
  --context_dim 512 \
  --transformer_depth 1 \
  --model_path $CKPT_PATH \
  --use_ddim True \
  --timestep_respacing ddim100 \
  $OPT