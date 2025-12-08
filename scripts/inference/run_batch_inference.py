import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image

# --- PATH SETUP ---
# Add repo root to path to allow imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(repo_root)

# Import Pipeline
from scripts.pipelines.pipeline_faceswap_inpaint import StableDiffusionIDControlInpaintPipeline

# Import Extractors (from your existing script)
# We wrap this in a try-except block or assume the script exists
try:
    from scripts.dataset.extract_all_conditions_single_image import (
        load_iresnet_model, 
        extract_embedding, 
        HRNetLandmarkDetector, 
        draw_landmarks,
        FacialMaskExtractor
    )
except ImportError:
    print("Error: Could not import extractors. Make sure 'scripts/dataset/extract_all_conditions_single_image.py' exists.")
    sys.exit(1)

# --- CONFIGURATION ---
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
# Paths to your trained models
CONTROLNET_PATH = "/users/erluo/scratch/controlnet-model/controlnet"
IP_ADAPTER_PATH = "/users/erluo/scratch/controlnet-model/ip_adapter/ip_adapter.bin"
FACEID_ENCODER_PATH = "checkpoints/glint360k_r100.pth"

# --- INPUTS ---
# 1. Target (Background) - Mandatory
TARGET_IMAGE_PATH = "assets/trump2.png"

# 2. Source (Identity) - Provide EITHER path to image OR path to precomputed embedding
SOURCE_IMAGE_PATH = "assets/trump1.png" 
PRECOMPUTED_EMBED_PATH = "" # e.g. "assets/trump1.pt". If exists, skips extraction.

# 3. Control (Landmarks) - Provide EITHER path to landmarks OR leave empty to compute from Target
PRECOMPUTED_LANDMARK_PATH = "" # e.g. "assets/trump2_landmarks.png"

# 4. Mask - Provide EITHER path to mask OR leave empty to compute from Target
PRECOMPUTED_MASK_PATH = "" # e.g. "assets/trump2_mask.png"

# Output
OUTPUT_PATH = "output_swap_auto.png"
DEVICE = "cuda"
DTYPE = torch.float16

def get_identity_embedding(source_path, embed_path, device):
    """Loads embedding if exists, else extracts from source image."""
    if embed_path and os.path.exists(embed_path):
        print(f"Loading precomputed ID embedding: {embed_path}")
        return torch.load(embed_path, map_location="cpu").to(dtype=DTYPE)
    
    print(f"Extracting ID embedding from: {source_path}")
    if not os.path.exists(source_path):
        raise ValueError(f"Source image not found: {source_path}")
        
    # Load Extractor
    iresnet, iresnet_device = load_iresnet_model(FACEID_ENCODER_PATH)
    
    # Extract
    embedding = extract_embedding(iresnet, iresnet_device, source_path)
    
    # Optional: Save it for next time
    # save_path = os.path.splitext(source_path)[0] + ".pt"
    # torch.save(embedding, save_path)
    
    return embedding.to(dtype=DTYPE)

def get_control_image(target_pil, landmark_path):
    """Loads landmark image if exists, else computes from target."""
    if landmark_path and os.path.exists(landmark_path):
        print(f"Loading precomputed landmarks: {landmark_path}")
        return load_image(landmark_path).resize((512, 512))
    
    print(f"Computing landmarks from Target...")
    detector = HRNetLandmarkDetector()
    landmarks = detector(target_pil)
    
    if landmarks is None:
        raise ValueError("No face detected in target for landmarks!")
        
    return draw_landmarks((512, 512), landmarks)

def get_mask_image(target_cv2, mask_path):
    """Loads mask if exists, else computes from target (with dilation)."""
    if mask_path and os.path.exists(mask_path):
        print(f"Loading precomputed mask: {mask_path}")
        return load_image(mask_path).resize((512, 512))
    
    print(f"Computing mask from Target...")
    extractor = FacialMaskExtractor()
    mask = extractor.extract_mask(target_cv2)
    
    if mask is None:
        raise ValueError("No face detected in target for masking!")
    
    # --- DILATION FIX ---
    kernel = np.ones((20, 20), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # --------------------
    
    return Image.fromarray(mask).convert("L")

def main():
    # 1. Load Pipeline
    print("Loading Pipeline...")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=DTYPE)
    pipe = StableDiffusionIDControlInpaintPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        ip_adapter_ckpt_path=IP_ADAPTER_PATH,
        faceid_embedding_dim=512,
        torch_dtype=DTYPE
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()

    # 2. Prepare Inputs
    if not os.path.exists(TARGET_IMAGE_PATH):
        raise ValueError(f"Target image missing: {TARGET_IMAGE_PATH}")

    target_pil = Image.open(TARGET_IMAGE_PATH).convert("RGB").resize((512, 512))
    target_cv2 = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)

    # A. Identity
    faceid_embeds = get_identity_embedding(SOURCE_IMAGE_PATH, PRECOMPUTED_EMBED_PATH, DEVICE)
    if faceid_embeds.dim() == 1: faceid_embeds = faceid_embeds.unsqueeze(0)

    # B. Landmarks
    control_image = get_control_image(target_pil, PRECOMPUTED_LANDMARK_PATH)

    # C. Mask
    mask_image = get_mask_image(target_cv2, PRECOMPUTED_MASK_PATH)

    # 3. Generate
    print("Running Inference...")
    result = pipe(
        prompt="high quality professional photo of a face",
        image=target_pil,       # Original Background
        mask_image=mask_image,  # Area to swap
        control_image=control_image, # Structure
        faceid_embeddings=faceid_embeds, # Identity
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        ip_adapter_scale=0.5
    ).images[0]
    
    result.save(OUTPUT_PATH)
    print(f"Saved result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()