import sys
import os
import argparse
import json
import torch
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import re

# ==================== PATH SETUP ====================
# Add project root to path to find 'pipelines'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import your custom pipeline
from scripts.pipelines.pipeline_faceswap import StableDiffusionIDControlPipeline

# Import your custom masking function
try:
    from scripts.dataset.extract_facial_mask import FacialMaskExtractor
except ImportError:
    print("Error: Could not import 'FacialMaskExtractor' from 'extract_facial_mask.py'.")
    sys.exit(1)

# ==================== CONFIGURATION ====================
DEFAULT_BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
DEFAULT_CONTROLNET = "/users/erluo/scratch/faceswap-diffusion/checkpoints/faceswap-model/checkpoint-30000/controlnet"
DEFAULT_IP_ADAPTER = "/users/erluo/scratch/faceswap-diffusion/checkpoints/faceswap-model/checkpoint-30000/ip_adapter/ip_adapter.bin"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def overlay_control_image_on_image(generated_image, control_image, alpha=0.5):
    """
    Overlay the pre-existing control image (landmarks) on top of the generated image.
    """
    # Resize control to match generated image if necessary
    if generated_image.size != control_image.size:
        control_image = control_image.resize(generated_image.size, Image.Resampling.LANCZOS)
    
    if generated_image.mode != 'RGBA':
        generated_image = generated_image.convert('RGBA')
    if control_image.mode != 'RGBA':
        control_image = control_image.convert('RGBA')
    
    # Blend
    overlaid = Image.blend(generated_image, control_image, alpha)
    return overlaid.convert('RGB')

def redact_gender_terms(prompt):
    if not prompt: return ""
    
    # Dictionary mapping target words to their neutral replacements
    replacements = {
        # Nouns
        "man": "person",
        "woman": "person",
        "girl": "person",
        "boy": "person",
        "guy": "person",
        "lady": "person",
        
        # Pronouns
        "he": "they",
        "she": "they",
        "him": "them",
        "his": "their",
        "her": "their", 
        "hers": "theirs"
    }
    
    redacted_prompt = prompt
    for target, replacement in replacements.items():
        # \b ensures we match whole words only
        pattern = re.compile(rf"\b{target}\b", re.IGNORECASE)
        redacted_prompt = pattern.sub(replacement, redacted_prompt)
        
    return redacted_prompt

def main():
    parser = argparse.ArgumentParser(description="Batch Face Swap Inference")
    
    # Inputs
    parser.add_argument("--config_json", required=True, help="Path to JSONL config file")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    
    # Model Configs
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--controlnet_path", default=DEFAULT_CONTROLNET)
    parser.add_argument("--ip_adapter_path", default=DEFAULT_IP_ADAPTER)
    
    # Generation Params
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    # 1. Load Models
    print(f"Loading ControlNet from {args.controlnet_path}...")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=DTYPE)
    
    print(f"Loading Pipeline...")
    pipe = StableDiffusionIDControlPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None
    )
    
    print(f"Loading IP-Adapter from {args.ip_adapter_path}...")
    pipe.load_ip_adapter_faceid(args.ip_adapter_path, image_emb_dim=512)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Initializing Face Mask Extractor...")
    mask_extractor = FacialMaskExtractor(True)
    
    # 2. Process Batch
    with open(args.config_json, 'r', encoding='utf-8') as f:
        try:
            items = json.load(f)
            if not isinstance(items, list):
                items = [items]
        except json.JSONDecodeError:
            f.seek(0)
            items = []
            for line in f:
                if line.strip():
                    items.append(json.loads(line))

    print(f"Found {len(items)} items to process.")

    for idx, item in enumerate(items):
        
        # --- A. Parse Paths from JSON ---
        target_path = item.get('target_path')
        control_path = item.get('control_path')
        source_img_path = item.get('source_path')
        raw_prompt = item.get('prompt')
        prompt = redact_gender_terms(raw_prompt)
        output_name = item.get('output_name', f"output_{idx}.png")
        
        # --- B. Derive Embedding Path ---
        base_source, _ = os.path.splitext(source_img_path)
        emb_path = base_source + ".pt"

        # --- C. Validation ---
        if not os.path.exists(target_path):
            print(f"Missing target: {target_path}")
            continue
        if not os.path.exists(control_path):
            print(f"Missing control: {control_path}")
            continue
        if not os.path.exists(emb_path):
            print(f"Missing embedding: {emb_path}")
            continue

        print(f"[{idx+1}/{len(items)}] Swapping {os.path.basename(source_img_path)} onto {os.path.basename(target_path)}")

        try:
            # --- D. Load Assets ---
            image_pil = load_image(target_path)   # Target (to be swapped)
            control_pil = load_image(control_path) # Landmarks
            
            faceid_embed = torch.load(emb_path, map_location="cpu")
            faceid_embed = faceid_embed.to(dtype=DTYPE)
            if faceid_embed.dim() == 1:
                faceid_embed = faceid_embed.unsqueeze(0)

            # --- E. Generate Mask On-The-Fly ---
            mask_np = mask_extractor.extract_mask_from_path(target_path)
            mask_pil = Image.fromarray(mask_np)

            # --- E-2. Save Generated Mask ---
            # Save mask to the same folder as control path (landmarks), but named after the TARGET image
            control_dir = os.path.dirname(control_path)
            
            # Use target filename for the mask name
            target_filename = os.path.basename(target_path)
            target_stem, _ = os.path.splitext(target_filename)
            mask_filename = f"{target_stem}_mask.png"
            
            mask_save_path = os.path.join(control_dir, mask_filename)
            mask_pil.save(mask_save_path)
            # print(f"Saved mask to: {mask_save_path}")
            
            # --- F. Inference ---
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            
            images = pipe(
                prompt=prompt,
                negative_prompt="noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
                image=image_pil,            # Source image (for inpainting context)
                mask_image=mask_pil,        # Generated mask
                control_image=control_pil,  # Pre-existing landmarks
                faceid_embeddings=faceid_embed,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                generator=generator
            ).images
            
            result_img = images[0]

            # --- G. Save Results ---
            # 1. Save Clean Result
            out_path = os.path.join(args.output_dir, output_name)
            result_img.save(out_path)
            
            # 2. Save Overlay (Result + Landmarks)
            base_out, ext = os.path.splitext(output_name)
            overlay_name = f"{base_out}_overlay{ext}"
            overlay_path = os.path.join(args.output_dir, overlay_name)
            
            overlay_img = overlay_control_image_on_image(result_img, control_pil, alpha=0.5)
            overlay_img.save(overlay_path)

        except Exception as e:
            print(f"Failed to process {target_path}: {e}")
            import traceback
            traceback.print_exc()

    print("Batch inference complete.")

if __name__ == "__main__":
    main()