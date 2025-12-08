#!/bin/bash

# source /etc/network_turbo
# export HF_HUB_DISABLE_XET=1

accelerate launch train_controlnet_ip-adapter.py \
    --pretrained_model_name_or_path "Manojb/stable-diffusion-2-1-base" \
    --conditioning_channels 3 \
    --train_data_dir "/root/autodl-tmp/ff-celeba-hq-dataset512-idcontrol" \
    --output_dir "/root/autodl-tmp/idcontrol-model" \
    --num_train_epochs 1 \
    --max_train_steps 100000 \
    --resolution 512 \
    --learning_rate 1e-5 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --checkpointing_steps 5000 \
    --checkpoints_total_limit 5 \
    --validation_steps 500 \
    --validation_prompt "A close-up photo of a person with light brown hair styled in loose waves, wearing a small earring, and a neutral expression." "A close-up photo of a baby with dark hair, wearing a green striped shirt, lying on a teal surface, looking directly at the camera with a calm expression." \
    --validation_image "/root/autodl-tmp/ff-celeba-hq-dataset512-idcontrol/landmarks/Part1/00087.png" "/root/autodl-tmp/ff-celeba-hq-dataset512-idcontrol/landmarks/Part1/00000.png" \
    --validation_faceid_embedding "/root/autodl-tmp/ff-celeba-hq-dataset512-idcontrol/embeddings/Part1/00087.pt" "/root/autodl-tmp/ff-celeba-hq-dataset512-idcontrol/embeddings/Part1/00000.pt" \
    --faceid_embedding_dim 512 \
    --ip_adapter_image_drop_rate 0.05 \
    --enable_ip_adapter \
    #--pretrained_ip_adapter_path "/path/to/pretrained_ip_adapter.bin" \
    #--use_fixed_timestep \

