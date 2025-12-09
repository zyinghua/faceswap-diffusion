#!/bin/bash
#SBATCH -p 3090-gcondo
#SBATCH -N 1
#SBATCH -n 4               
#SBATCH --gres=gpu:4      
#SBATCH --mem=96G         
#SBATCH -t 12:00:00
#SBATCH -J turbo_4gpu
#SBATCH -o logs/train_4gpu_%j.out
#SBATCH -e logs/train_4gpu_%j.err

module load anaconda
module load cuda/11.8

source activate /users/erluo/scratch/faceswap_env

export HF_HOME="/users/erluo/scratch/hf_cache"
export HF_DATASETS_CACHE="/users/erluo/scratch/hf_cache/datasets"
mkdir -p $HF_DATASETS_CACHE

# 1. INCREASE TIMEOUT 
export NCCL_TIMEOUT=7200
export NCCL_BLOCKING_WAIT=1

# 2. Network Stability for 3090s
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# ------------------------------------------------------------------

accelerate launch \
    --main_process_port $MASTER_PORT \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision "fp16" \
    --dynamo_backend "no" \
    --multi_gpu \
    train_controlnet.py \
    --pretrained_model_name_or_path "stabilityai/sd-turbo" \
    --conditioning_channels 3 \
    --train_data_dir "/users/erluo/scratch/canny_dataset" \
    --output_dir "/users/erluo/scratch/controlnet-model" \
    --num_train_epochs 1 \
    --max_train_steps 20000 \
    --resolution 512 \
    --learning_rate 1e-5 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 5 \
    --validation_steps 500 \
    --validation_prompt "A close-up photo of a person with light brown hair styled in loose waves, wearing a small earring, and a neutral expression." "A close-up photo of a baby with dark hair, wearing a green striped shirt, lying on a teal surface, looking directly at the camera with a calm expression." \
    --validation_image "/users/erluo/scratch/canny_dataset/canny/Part1/00087.png" "/users/erluo/scratch/canny_dataset/canny/Part1/00000.png" \
    --use_fixed_timestep \
    --report_to="wandb"

