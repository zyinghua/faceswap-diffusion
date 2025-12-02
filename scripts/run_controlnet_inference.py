# Template from diffusers: https://github.com/huggingface/diffusers/tree/main/examples/controlnet

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.pipeline_controlnet import StableDiffusionControlNetPipeline
from models.controlnet import ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import random
import json

BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
CONTROLNET_PATH = "" # pretrained controlnet path, ends with /controlnet
PROMPT_JSONL_PATH = ""  # Path to JSONL file containing prompts (each line should have "text")
CONTROL_IMAGE = "" # canny image
NUM_INFERENCE_STEPS = 20
OUTPUT_PATH = "./output.png"
SEED = random.randint(0, 1000000)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.bfloat16

# Guidance parameter (default: 7.5):
GUIDANCE_SCALE = 7.5
# controlnet_conditioning_scale: Controls how strongly the model follows the CONTROL IMAGE (default: 1.0) 
# - Empirical finding: do not change from 1.0.
CONTROLNET_CONDITIONING_SCALE = 1.0


def main():
    control_image_filename = CONTROL_IMAGE.split("/")[-1]
    
    prompt = None
    if PROMPT_JSONL_PATH:
        with open(PROMPT_JSONL_PATH, 'r', encoding='utf-8') as f:
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
        raise ValueError(f"Could not find prompt for {control_image_filename} in {PROMPT_JSONL_PATH}")
    
    print(f"Loading ControlNet from: {CONTROLNET_PATH}")
    print(f"Prompt: {prompt}")
    
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=DTYPE
    )
    
    # Speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimization
    pipe.enable_model_cpu_offload()

    control_image = load_image(CONTROL_IMAGE)
    
    # Generate image
    generator = torch.manual_seed(SEED)
    image = pipe(
        prompt, 
        num_inference_steps=NUM_INFERENCE_STEPS, 
        generator=generator, 
        image=control_image,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE
    ).images[0]
    
    image.save(OUTPUT_PATH)


if __name__ == "__main__":
    main()