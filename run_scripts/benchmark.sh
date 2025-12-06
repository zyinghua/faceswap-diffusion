#!/bin/bash

python ../scripts/inference/benchmark_controlnets.py \
  --captions_json /users/erluo/scratch/captions.jsonl \
  --dataset_root /users/erluo/scratch/canny_dataset \
  --output_dir /users/erluo/scratch/benchmark_results \
  --num_samples 5 \
  --ckpt_short /users/erluo/scratch/faceswap-diffusion/checkpoints/comparison/short/checkpoint-15000/controlnet \
  --ckpt_medium /users/erluo/scratch/faceswap-diffusion/checkpoints/comparison/medium/checkpoint-15000/controlnet \
  --ckpt_generic /users/erluo/scratch/faceswap-diffusion/checkpoints/comparison/generic/checkpoint-10000/controlnet

