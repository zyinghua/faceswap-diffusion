import sys
import os
import argparse
import json
import torch
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# --- Add repo root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.pipelines.pipeline_faceswap import StableDiffusionIDControlPipeline

# Import extractors
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
    if not prompt: return ""
    targets = ["man", "woman", "girl", "boy"]
    redacted_prompt = prompt
    for target in targets:
        pattern = re.compile(rf"\b{target}\b", re.IGNORECASE)
        redacted_prompt = pattern.sub("person", redacted_prompt)
    return redacted_prompt

def save_visual_comparison(save_path, src_img_path, tgt_pil, landmarks_pil, result_pil, prompt):
    src_pil = Image.open(src_img_path).convert("RGB").resize((512, 512))
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    def set_ax(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis("off")

    set_ax(axes[0], src_pil, "Source Identity")
    set_ax(axes[1], tgt_pil, "Target Body")
    set_ax(axes[2], landmarks_pil, "Landmarks (Control)")
    set_ax(axes[3], result_pil, "Swapped Result")
    
    wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=80))
    fig.suptitle(f"Prompt: {wrapped_prompt}", fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def run_batch(args):
    # 1. Load Models
    print("--- Loading Pipeline Models ---")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=DTYPE)
    
    pipe = StableDiffusionIDControlPipeline.from_pretrained(
        BASE_MODEL, 
        controlnet=controlnet, 
        torch_dtype=DTYPE, 
        safety_checker=None
    )
    
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
    images_dir = output_dir / "images"
    visuals_dir = output_dir / "visuals"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Inference on {len(batch_list)} items ---")
    
    for item in tqdm(batch_list):
        try:
            # Parse Item
            src_path = Path(item["source_path"])
            tgt_path = Path(item["target_path"])
            out_name = item["output_name"]
            
            raw_prompt = item.get("prompt", "a photo of a person")
            final_prompt = redact_gender_terms(raw_prompt)
            
            save_path = images_dir / out_name
            visual_path = visuals_dir / f"vis_{out_name}"
            
            if save_path.exists(): continue 

            # --- A. Get FaceID Embedding (FIXED) ---
            src_embed_path = src_path.with_suffix('.pt')
            
            if src_embed_path.exists():
                faceid_embed = torch.load(src_embed_path, map_location="cpu")
            else:
                faceid_embed = extract_embedding(iresnet, iresnet_device, str(src_path))
                torch.save(faceid_embed.cpu(), src_embed_path)

            # Defensive Shape Normalization
            if isinstance(faceid_embed, np.ndarray):
                faceid_embed = torch.from_numpy(faceid_embed)
            
            faceid_embed = faceid_embed.to(device=DEVICE, dtype=DTYPE)
            
            # Flatten to [512] then reshape to [1, 512]
            # This handles both [512], [1, 512] and even [1, 1, 512] inputs safely
            faceid_embed = faceid_embed.view(1, -1)

            # --- B. Get Target Inputs ---
            tgt_pil = Image.open(tgt_path).convert("RGB").resize((512, 512))

            # --- C. Get Landmarks ---
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

            # --- D. Inference ---
            images = pipe(
                prompt=final_prompt,
                negative_prompt="noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
                image=None, 
                control_image=control_image,
                # NOTE: We pass the tensor directly now, NO .unsqueeze(0) here
                faceid_embeddings=faceid_embed, 
                num_inference_steps=50,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                ip_adapter_scale=None
            ).images

            # Save Result
            result_img = images[0]
            result_img.save(save_path)
            
            # Save Visualization
            save_visual_comparison(
                visual_path, 
                src_path, 
                tgt_pil, 
                control_image, 
                result_img, 
                final_prompt
            )
            
        except Exception as e:
            print(f"Failed on {out_name}: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--ip_adapter_path", type=str, required=True)
    parser.add_argument("--faceid_encoder_path", type=str, default="checkpoints/glint360k_r100.pth")
    args = parser.parse_args()
    
    run_batch(args)