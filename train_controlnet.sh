#!/bin/bash

accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path "stabilityai/sd-turbo" \
    --conditioning_channels 3 \
    --train_data_dir "ffhq-dataset512-canny" \
    --output_dir "controlnet-model" \
    --num_train_epochs 10 \
    --resolution 512 \
    --learning_rate 1e-5 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 1000 \
    --checkpoints_total_limit 5 \
    --validation_steps 500 \
    --validation_prompt "high-quality professional photo of a face" "high-quality professional photo of a face" "high-quality professional photo of a face" \
    --validation_image "ffhq-dataset512-canny/canny/Part2/17090.png" "ffhq-dataset512-canny/canny/Part7/65173.png" "ffhq-dataset512-canny/canny/Part5/41191.png" \
    --num_validation_images 1 \