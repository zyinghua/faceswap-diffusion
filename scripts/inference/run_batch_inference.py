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

# --- Add repo root to path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.pipelines.pipeline_faceswap_inpaint import StableDiffusionIDControlInpaintPipeline

# Import your condition extractors
try:
    from scripts.dataset.extract_all_conditions_single_image import (
        load_iresnet_model, extract_embedding, 
        HRNetLandmarkDetector, draw_landmarks, 
        FacialMaskExtractor
    )
except ImportError:
    print("Could not import extractors. Make sure extract_all_conditions_single_image.py exists.")
    sys.exit(1)

# --- CONFIGURATION ---
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def run_batch(args):
    # 1. Load Models (Once)
    print("--- Loading Pipeline Models ---")
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=DTYPE)
    pipe = StableDiffusionIDControlInpaintPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, ip_adapter_ckpt_path=args.ip_adapter_path, 
        faceid_embedding_dim=512, torch_dtype=DTYPE, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()

    # 2. Load Extractors (Once)
    print("--- Loading Extractor Models ---")
    iresnet, iresnet_device = load_iresnet_model(args.faceid_encoder_path)
    landmark_detector = HRNetLandmarkDetector()
    mask_extractor = FacialMaskExtractor()

    # 3. Load Config
    with open(args.config_json, 'r') as f:
        batch_list = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Inference on {len(batch_list)} pairs ---")
    
    for item in tqdm(batch_list):
        try:
            # Paths
            src_path = Path(item["source_path"])
            tgt_path = Path(item["target_path"])
            out_name = item["output_name"]
            save_path = output_dir / out_name
            
            if save_path.exists(): continue # Skip if already done

            # --- A. Get FaceID Embedding (Lazy Load) ---
            # Look for source.pt
            src_embed_path = src_path.with_suffix('.pt')
            
            if src_embed_path.exists():
                faceid_embed = torch.load(src_embed_path, map_location="cpu").to(dtype=DTYPE)
            else:
                # Extract and Save
                faceid_embed = extract_embedding(iresnet, iresnet_device, str(src_path)).to(dtype=DTYPE)
                torch.save(faceid_embed.cpu(), src_embed_path)

            # --- B. Get Target Inputs (Image) ---
            tgt_pil = Image.open(tgt_path).convert("RGB").resize((512, 512))
            tgt_cv2 = cv2.cvtColor(np.array(tgt_pil), cv2.COLOR_RGB2BGR)

            # --- C. Get Landmarks (Lazy Load) ---
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

            # --- D. Get Mask (Lazy Load) ---
            mask_path = tgt_path.parent / (tgt_path.stem + "_mask.png")
            
            if mask_path.exists():
                mask_pil = load_image(str(mask_path)).resize((512, 512))
            else:
                mask = mask_extractor.extract_mask(tgt_cv2)
                if mask is None:
                    print(f"Skip {out_name}: No face for mask")
                    continue
                
                # Dilate Mask (The Fix for seams)
                kernel = np.ones((20, 20), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask_pil = Image.fromarray(mask).convert("L")
                mask_pil.save(mask_path)

            # --- E. Inference ---
            image = pipe(
                prompt="high quality professional photo of a face",
                image=tgt_pil,
                mask_image=mask_pil,
                control_image=control_image,
                faceid_embeddings=faceid_embed.unsqueeze(0),
                num_inference_steps=30,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                ip_adapter_scale=0.5
            ).images[0]
            
            image.save(save_path)
            
        except Exception as e:
            print(f"Failed on {out_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", type=str, required=True, help="Path to inference_config.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save swaps")
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--ip_adapter_path", type=str, required=True)
    parser.add_argument("--faceid_encoder_path", type=str, default="checkpoints/glint360k_r100.pth")
    args = parser.parse_args()
    
    run_batch(args)