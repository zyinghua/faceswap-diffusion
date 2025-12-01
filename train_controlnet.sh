#!/bin/bash

#export HF_HUB_DISABLE_XET=1

accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path "Manojb/stable-diffusion-2-1-base" \
    --conditioning_channels 3 \
    --train_data_dir "/root/autodl-tmp/ffhq-dataset512-canny" \
    --output_dir "/root/autodl-tmp/controlnet-model" \
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
    --validation_image "/root/autodl-tmp/ffhq-dataset512-canny/canny/Part1/00087.png" "/root/autodl-tmp/ffhq-dataset512-canny/canny/Part1/00000.png" \
    # --use_fixed_timestep \