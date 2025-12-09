# Template from diffusers: https://github.com/huggingface/diffusers/tree/main/examples/controlnet

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import json
from PIL import Image

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
CONTROLNET_PATH = "" # pretrained controlnet path, ends with /controlnet
PROMPT = None
METADATA_JSONL_PATH = ""  # Path to JSONL file containing prompts (each line should have "text")
CONTROL_IMAGE = "" # control image (e.g., canny edges, landmarks, etc.)
NUM_INFERENCE_STEPS = 30
OUTPUT_PATH = "./generated_images"
SEED = None
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16
NEGATIVE_PROMPT = "noisy, blurry, low contrast, fade, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
SAMPLE_NUM = 10

# Guidance parameter (default: 7.5):
GUIDANCE_SCALE = 7.5
# controlnet_conditioning_scale: Controls how strongly the model follows the CONTROL IMAGE (default: 1.0) 
# - Empirical finding: do not change from 1.0.
CONTROLNET_CONDITIONING_SCALE = 1.0

# Overlay parameters
ENABLE_CONTROL_IMAGE_OVERLAY = True
CONTROL_IMAGE_OVERLAY_ALPHA = 0.5


def overlay_control_image_on_image(generated_image, control_image, alpha=0.5):
    """
    Overlay the control image on top of the generated image.
    """
    if generated_image.size != control_image.size:
        control_image = control_image.resize(generated_image.size, Image.Resampling.LANCZOS)
    
    # Convert to RGBA if needed
    if generated_image.mode != 'RGBA':
        generated_image = generated_image.convert('RGBA')
    if control_image.mode != 'RGBA':
        control_image = control_image.convert('RGBA')
    
    # Blend the images
    overlaid = Image.blend(generated_image, control_image, alpha)
    
    return overlaid.convert('RGB')


def main():
    # Get prompt from metadata or use provided prompt
    prompt = PROMPT
    if not prompt and METADATA_JSONL_PATH:
        control_image_filename = os.path.basename(CONTROL_IMAGE)
        with open(METADATA_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                file_name = entry.get('file_name', '')
                if file_name.endswith(control_image_filename):
                    prompt = entry.get('text', None)
                    break
    
    if not prompt:
        raise ValueError(f"Prompt must be provided either via PROMPT variable or found in {METADATA_JSONL_PATH}")
    
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
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
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
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Save images
    for i, image in enumerate(images):
        original_path = os.path.join(OUTPUT_PATH, f"generated_image{i}.png")
        image.save(original_path)
        print(f"Saved original image to: {original_path}")
        
        # Create and save control image overlaid version if overlay is enabled
        if ENABLE_CONTROL_IMAGE_OVERLAY:
            overlaid_image = overlay_control_image_on_image(image, control_image, alpha=CONTROL_IMAGE_OVERLAY_ALPHA)
            overlaid_path = os.path.join(OUTPUT_PATH, f"generated_image{i}_with_control_overlay.png")
            overlaid_image.save(overlaid_path)
            print(f"Saved control image overlaid image to: {overlaid_path}")


if __name__ == "__main__":
    main()