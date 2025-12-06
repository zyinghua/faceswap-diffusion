import argparse
import json
import shutil
import os
import random
import sys
import numpy as np
import cv2
import torch
import facer
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. Landmark Detector (HRNet)
# ==========================================
class HRNetLandmarkDetector:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading HRNet Landmark Detector on {device}...")
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=self.device)
        self.landmark_detector = facer.face_aligner('farl/wflw/448', device=self.device)

    def __call__(self, image):
        # Facer expects [B, C, H, W] tensor with values 0-255
        img_tensor = TF.to_tensor(image).to(self.device).unsqueeze(0) * 255.0
        
        with torch.inference_mode():
            faces = self.face_detector(img_tensor)
            if 'image_ids' not in faces or len(faces['image_ids']) == 0:
                return None
            faces = self.landmark_detector(img_tensor, faces)
            
        return faces['alignment'][0].cpu().numpy()

def draw_landmarks(image_size, landmarks):
    """Draws 98-point WFLW landmarks as DOTS (circles)."""
    H, W = image_size
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    if landmarks is None:
        return Image.fromarray(canvas)

    pts = landmarks.astype(np.int32)

    def draw_points(indices, color):
        valid_indices = [i for i in indices if i < len(pts)]
        for i in valid_indices:
            x, y = pts[i]
            cv2.circle(canvas, (x, y), 3, color, -1)

    # WFLW 98-point Mapping
    draw_points(range(0, 33), (255, 255, 255)) # Jaw
    draw_points(range(33, 42), (255, 255, 0))  # Left Brow
    draw_points(range(42, 51), (255, 255, 0))  # Right Brow
    draw_points(range(51, 55), (255, 0, 255))  # Nose Bridge
    draw_points(range(55, 60), (255, 0, 255))  # Nose Tip
    
    if 76 < len(pts):
        draw_points(range(60, 68), (0, 255, 0))   # Left Eye
        draw_points(range(68, 76), (0, 255, 0))   # Right Eye
    
    if 97 < len(pts):
        draw_points(range(76, 88), (0, 0, 255))   # Outer Mouth
        draw_points(range(88, 98), (0, 0, 255))   # Inner Mouth

    return Image.fromarray(canvas)

def get_caption(captions_dict, rel_path_str, filename, default_caption):
    """Robust caption lookup."""
    if rel_path_str in captions_dict:
        return captions_dict[rel_path_str]
    elif filename in captions_dict:
        return captions_dict[filename]
    return default_caption

def process_images_recursive(input_dir, output_dir, 
                             captions_json,
                             embeddings_dir,
                             landmark_subfolder="landmarks", 
                             embedding_subfolder="embeddings",
                             source_generic_caption="high-quality close-up photo of a face",
                             target_generic_caption=None,
                             max_samples=None,
                             is_faceswap=False):
    
    input_path = Path(input_dir)
    embeddings_input_path = Path(embeddings_dir)
    output_path = Path(output_dir)
    landmark_path = output_path / landmark_subfolder
    embed_path = output_path / embedding_subfolder
    
    output_path.mkdir(parents=True, exist_ok=True)
    landmark_path.mkdir(parents=True, exist_ok=True)
    embed_path.mkdir(parents=True, exist_ok=True)

    # --- Load Captions (only if target_generic_caption is None) ---
    captions_dict = {}
    if target_generic_caption is None:
        print(f"Loading captions from {captions_json}...")
        with open(captions_json, 'r', encoding='utf-8') as f:
            captions_dict = json.load(f)
        print(f"Loaded {len(captions_dict)} captions.")
    else:
        print(f"Using target_generic_caption, skipping captions.json")
    
    # Gather images
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images total.")
    
    if max_samples is not None:
        print(f"Running on first {max_samples} images for inspection...")
        image_files = image_files[:max_samples]
    
    # 1. Initialize Landmark Detector
    landmark_detector = HRNetLandmarkDetector()
    
    # 2. Build Identity Map for FaceSwap pairing
    identity_map = {}
    if is_faceswap:
        for img_file in image_files:
            parent = str(img_file.parent)
            if parent not in identity_map:
                identity_map[parent] = []
            identity_map[parent].append(img_file)

    metadata_entries = []
    processed_count = 0
    missing_caption_count = 0
    missing_embedding_count = 0
    
    for img_file in tqdm(image_files, desc="Processing dataset"):
        try:
            # Calculate paths relative to input_dir
            rel_path = img_file.relative_to(input_path)
            rel_path_str = str(rel_path)
            
            # 1. Look up Caption
            if target_generic_caption is not None:
                current_caption = target_generic_caption
            else:
                current_caption = get_caption(captions_dict, rel_path_str, img_file.name, None)
                if current_caption is None:
                    missing_caption_count += 1
                    continue

            # 2. Look up Input Embedding
            # Assumes embedding has same relative path but with .pt extension
            input_embed_file = embeddings_input_path / rel_path.with_suffix('.pt')
            if not input_embed_file.exists():
                missing_embedding_count += 1
                continue

            # Paths for Output
            original_output = output_path / rel_path
            landmark_output = landmark_path / rel_path.with_suffix('.png')
            embed_output = embed_path / rel_path.with_suffix('.pt')
            
            # Create subdirs
            original_output.parent.mkdir(parents=True, exist_ok=True)
            landmark_output.parent.mkdir(parents=True, exist_ok=True)
            embed_output.parent.mkdir(parents=True, exist_ok=True)

            # --- A. Copy Original Image ---
            if not original_output.exists():
                shutil.copy2(img_file, original_output)
            
            # --- B. Copy Pre-computed Embedding ---
            if not embed_output.exists():
                shutil.copy2(input_embed_file, embed_output)

            # --- C. Generate Landmarks (HRNet) ---
            # Only run if not exists to save time
            if not landmark_output.exists():
                image = Image.open(img_file).convert("RGB")
                landmarks = landmark_detector(image)
                
                if landmarks is None:
                    continue # Skip if no face for landmarks

                landmark_image = draw_landmarks((image.height, image.width), landmarks)
                landmark_image.save(landmark_output)

            # --- D. Metadata Generation ---
            rel_img_str = rel_path_str
            rel_cond_str = str(Path(landmark_subfolder) / rel_path.with_suffix('.png'))
            rel_embed_str = str(Path(embedding_subfolder) / rel_path.with_suffix('.pt'))
            
            if not is_faceswap:
                entry = {
                    "file_name": rel_img_str,
                    "text": current_caption,
                    "conditioning_image": rel_cond_str,
                    "faceid_embedding": rel_embed_str
                }
            else:
                # FaceSwap Format
                parent = str(img_file.parent)
                siblings = identity_map.get(parent, [])
                
                # Pick Source Image
                source_candidates = [s for s in siblings if s != img_file]
                if len(source_candidates) > 0:
                    source_file = random.choice(source_candidates)
                else:
                    source_file = img_file
                
                # Get Source Paths
                source_rel_path = source_file.relative_to(input_path)
                
                # Look up Source Embedding Input
                source_input_embed = embeddings_input_path / source_rel_path.with_suffix('.pt')
                
                # Define Source Embedding Output
                source_embed_output = embed_path / source_rel_path.with_suffix('.pt')
                
                # Copy Source Embedding if needed
                if not source_embed_output.exists():
                    if source_input_embed.exists():
                        source_embed_output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_input_embed, source_embed_output)
                    else:
                        # Fallback to target embedding if source missing
                        source_embed_output = embed_output 
                        source_file = img_file

                source_embed_str = str(Path(embedding_subfolder) / source_file.relative_to(input_path).with_suffix('.pt'))

                entry = {
                    "target_img": rel_img_str,
                    "caption": current_caption,
                    "target_img_landmarks": rel_cond_str,
                    "target_img_encoding": rel_embed_str,
                    "source_img_encoding": source_embed_str,
                    "source_img_caption": source_generic_caption
                }

            metadata_entries.append(entry)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    # Save metadata
    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\nProcessed {processed_count} images.")
    if missing_caption_count > 0:
        print(f"Warning: Skipped {missing_caption_count} images due to missing captions.")
    if missing_embedding_count > 0:
        print(f"Warning: Skipped {missing_embedding_count} images due to missing pre-computed embeddings.")
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--captions_json", type=str, required=False, help="Path to JSON file with captions")
    parser.add_argument("--embeddings_dir", type=str, required=True, help="Path to directory with pre-computed .pt embeddings")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--source_generic_caption", type=str, default="high-quality close-up photo of a face")
    parser.add_argument("--target_generic_caption", type=str, default=None, help="If provided, use this for all target captions and ignore captions.json")
    parser.add_argument("--faceswap", action="store_true", help="Enable FaceSwap tuple output format")
    
    args = parser.parse_args()
    
    # Validate: if target_generic_caption is None, captions_json must be provided
    if args.target_generic_caption is None and args.captions_json is None:
        parser.error("--captions_json is required when --target_generic_caption is not provided")
    
    process_images_recursive(
        args.input_dir,
        args.output_dir,
        args.captions_json,
        args.embeddings_dir,
        source_generic_caption=args.source_generic_caption,
        target_generic_caption=args.target_generic_caption,
        max_samples=args.max_samples,
        is_faceswap=args.faceswap
    )

if __name__ == "__main__":
    main()