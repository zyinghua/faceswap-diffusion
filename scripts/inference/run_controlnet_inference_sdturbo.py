# Template from diffusers: https://github.com/huggingface/diffusers/tree/main/examples/controlnet

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerDiscreteScheduler
from diffusers.utils import load_image
import torch
import json

BASE_MODEL = "stabilityai/sd-turbo"
CONTROLNET_PATH = "/users/erluo/scratch/faceswap-diffusion/checkpoints/comparison/landmark-controlnet/checkpoint-25000/controlnet" # pretrained controlnet path, ends with /controlnet
METADATA_JSONL_PATH = ""  # Path to JSONL file containing prompts (each line should have "text")
CONTROL_IMAGE = "/users/erluo/scratch/faceswap-diffusion/scripts/evaluation/eval_images/img3/69051_landmark.png" # canny image
NUM_INFERENCE_STEPS = 2
OUTPUT_PATH = "/users/erluo/scratch/faceswap-diffusion/assets/landmark_control/" # path to output dir
SEED = None
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
NEGATIVE_PROMPT = "faded, noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
SAMPLE_NUM = 10

# Guidance parameter (default: 7.5):
GUIDANCE_SCALE = 0.5
# controlnet_conditioning_scale: Controls how strongly the model follows the CONTROL IMAGE (default: 1.0) 
# - Empirical finding: do not change from 1.0.
CONTROLNET_CONDITIONING_SCALE = 1.0


def main():
    control_image_filename = CONTROL_IMAGE.split("/")[-1]
    
    prompt = "A close-up photo of a woman with dark hair styled back, wearing a pearl earring, and applying makeup to her eye with a soft smile."
    if METADATA_JSONL_PATH:
        with open(METADATA_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                file_name = entry.get('file_name', '')
                if file_name.endswith(control_image_filename):
                    prompt = entry.get('text')
                    break
    
    if not prompt:
        raise ValueError(f"Could not find prompt for {control_image_filename} in {METADATA_JSONL_PATH}")
    
    print(f"Loading ControlNet from: {CONTROLNET_PATH}")
    print(f"Prompt: {prompt}")
    
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=DTYPE,
        safety_checker=None,  # Disable safety checker for faster inference
    )
    
    # Speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimization
    pipe.enable_model_cpu_offload()

    control_image = load_image(CONTROL_IMAGE)

    prompts = [prompt] * SAMPLE_NUM
    negative_prompts = [NEGATIVE_PROMPT] * SAMPLE_NUM
    
    # Generate image
    if SEED is not None:
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        print(f"Using seed: {SEED}")
    else:
        generator = None
    
    images = pipe(
        prompts,
        negative_prompt=negative_prompts,
        num_inference_steps=NUM_INFERENCE_STEPS, 
        generator=generator, 
        image=control_image,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE
    ).images
    
    for i, image in enumerate(images):
        image.save(OUTPUT_PATH + f"generated_image{i}.png")


if __name__ == "__main__":
    main()