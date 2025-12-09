import sys
import os
import argparse
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import re

# --- Add repo root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# NOTE: We switched back to the standard pipeline, not the Inpaint one
from pipelines.pipeline_faceswap import StableDiffusionIDControlPipeline

# Import extractors (Only for Embedding and Landmarks now)
try:
    from scripts.dataset.extract_all_conditions_single_image import (
        load_iresnet_model, extract_embedding, 
        HRNetLandmarkDetector, draw_landmarks
    )
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def redact_gender_terms(prompt):
    """
    Replaces whole words 'man', 'woman', 'girl', 'boy' with 'person'.
    Case-insensitive. Preserves surrounding text.
    """
    if not prompt:
        return ""
        
    # List of words to replace
    targets = ["man", "woman", "girl", "boy"]
    
    redacted_prompt = prompt
    
    for target in targets:
        # Regex explanation:
        # \b       : Word boundary (ensures we don't match inside 'batman')
        # {target} : The word we are looking for
        # \b       : Word boundary
        # (?i)     : Case insensitive flag (handled by re.IGNORECASE)
        pattern = re.compile(rf"\b{target}\b", re.IGNORECASE)
        
        # Replace found instance with "person"
        redacted_prompt = pattern.sub("person", redacted_prompt)
        
    return redacted_prompt

def run_batch(args):
    # 1. Load Models
    print("--- Loading Pipeline Models ---")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=DTYPE)
    
    # Using the standard ID Control Pipeline (No Inpainting)
    pipe = StableDiffusionIDControlPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=DTYPE, 
        safety_checker=None
    )
    
    # Load IP Adapter
    pipe.load_ip_adapter_faceid(args.ip_adapter_path, image_emb_dim=512)
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()

    # 2. Load Extractors
    print("--- Loading Extractor Models ---")
    iresnet, iresnet_device = load_iresnet_model(args.faceid_encoder_path)
    landmark_detector = HRNetLandmarkDetector()
    
    # 3. Load Config
    with open(args.config_json, 'r') as f:
        batch_list = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Inference on {len(batch_list)} items ---")
    
    for item in tqdm(batch_list):
        try:
            # Parse Item
            src_path = Path(item["source_path"])
            tgt_path = Path(item["target_path"])
            out_name = item["output_name"]
            
            # Get Prompt and Identity Name for redaction
            if "prompt" in item:
                raw_prompt = item["prompt"]
            else:
                raise ValueError(f"Missing 'prompt' in item: {item}")
            
            # Redact Prompt
            final_prompt = redact_prompt(raw_prompt, identity_name)
            
            save_path = output_dir / out_name
            if save_path.exists(): continue 

            # --- A. Get FaceID Embedding ---
            src_embed_path = src_path.with_suffix('.pt')
            if src_embed_path.exists():
                faceid_embed = torch.load(src_embed_path, map_location="cpu").to(dtype=DTYPE)
            else:
                faceid_embed = extract_embedding(iresnet, iresnet_device, str(src_path)).to(dtype=DTYPE)
                torch.save(faceid_embed.cpu(), src_embed_path)

            # --- B. Get Target Inputs ---
            tgt_pil = Image.open(tgt_path).convert("RGB").resize((512, 512))

            # --- C. Get Landmarks (Control Image) ---
            landmark_path = tgt_path.parent / (tgt_path.stem + "_landmarks.png")
            if landmark_path.exists():
                control_image = load_image(str(landmark_path)).resize((512, 512))
            else:
                landmarks = landmark_detector(tgt_pil)
                if landmarks is None: 
                    print(f"Skip {out_name}: No face for landmarks")
                    continue
                control_image = draw_landmarks((512, 512), landmarks)
                control_image.save(landmark_path)

            # --- D. Inference (No Mask) ---
            # Note: We pass valid prompts now. 
            # We do NOT pass mask_image.
            images = pipe(
                prompt=final_prompt,
                negative_prompt="faded, noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
                image=None, # Set to None for pure generation, or tgt_pil for Img2Img style if pipeline supports it
                control_image=control_image,
                faceid_embeddings=faceid_embed.unsqueeze(0),
                num_inference_steps=50,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                ip_adapter_scale=None
            ).images

            images[0].save(save_path)
            
        except Exception as e:
            print(f"Failed on {out_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--ip_adapter_path", type=str, required=True)
    parser.add_argument("--faceid_encoder_path", type=str, default="checkpoints/glint360k_r100.pth")
    args = parser.parse_args()
    
    run_batch(args)