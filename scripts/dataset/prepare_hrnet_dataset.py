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
from insightface.app import FaceAnalysis

# --- IMPORT FROM SIBLING SCRIPT ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from insightface_app_embedding_extraction import process_single_image
except ImportError:
    print("Error: Could not import 'insightface_app_embedding_extraction.py'.")
    print("Make sure both scripts are in the same directory.")
    sys.exit(1)

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

def process_images_recursive(input_dir, output_dir, 
                             captions_json,
                             landmark_subfolder="landmarks", 
                             embedding_subfolder="embeddings",
                             default_caption="high-quality professional photo of a face", 
                             max_samples=None,
                             is_faceswap=False):
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    landmark_path = output_path / landmark_subfolder
    embed_path = output_path / embedding_subfolder
    
    output_path.mkdir(parents=True, exist_ok=True)
    landmark_path.mkdir(parents=True, exist_ok=True)
    embed_path.mkdir(parents=True, exist_ok=True)

    # --- Load Captions ---
    print(f"Loading captions from {captions_json}...")
    with open(captions_json, 'r', encoding='utf-8') as f:
        captions_dict = json.load(f)
    print(f"Loaded {len(captions_dict)} captions.")

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
    
    # 2. Initialize InsightFace
    print("Loading InsightFace (AntelopeV2)...")
    app = FaceAnalysis(name="antelopev2", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # 3. Build Identity Map for FaceSwap pairing
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
    
    for img_file in tqdm(image_files, desc="Processing dataset"):
        try:
            rel_path = img_file.relative_to(input_path)
            rel_path_str = str(rel_path)

            # Check for Caption (Skip if missing, like the canny script)
            if rel_path_str not in captions_dict:
                missing_caption_count += 1
                continue
            
            current_caption = captions_dict[rel_path_str]
            
            # Paths
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
            
            # --- B. Generate Landmarks (HRNet) ---
            image = Image.open(img_file).convert("RGB")
            landmarks = landmark_detector(image)
            
            if landmarks is None:
                continue # Skip if no face for landmarks

            landmark_image = draw_landmarks((image.height, image.width), landmarks)
            landmark_image.save(landmark_output)
            
            # --- C. Generate ID Embedding (Using other script) ---
            success, _ = process_single_image(app, img_file, embed_output)
            if not success:
                continue # Skip if InsightFace fails to find a face

            # --- D. Metadata Generation ---
            rel_img_str = str(rel_path)
            rel_cond_str = str(Path(landmark_subfolder) / rel_path.with_suffix('.png'))
            rel_embed_str = str(Path(embedding_subfolder) / rel_path.with_suffix('.pt'))
            
            if not is_faceswap:
                # Standard ControlNet Format
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
                    # Fallback to self
                    source_file = img_file
                
                # Get Source Metadata
                source_rel_path = source_file.relative_to(input_path)
                source_rel_path_str = str(source_rel_path)
                
                # Get Source Caption default_cation
                source_caption = default_caption
                
                # Get Source Embedding Path
                source_embed_output = embed_path / source_rel_path.with_suffix('.pt')
                
                if not source_embed_output.exists():
                    # Generate it on demand
                    source_embed_output.parent.mkdir(parents=True, exist_ok=True)
                    s_success, _ = process_single_image(app, source_file, source_embed_output)
                    if not s_success:
                        # If source fails, fallback to target (self) which we know works
                        source_file = img_file
                        source_embed_output = embed_output
                        source_caption = current_caption
                
                source_embed_str = str(Path(embedding_subfolder) / source_file.relative_to(input_path).with_suffix('.pt'))

                # Exact Output Format
                entry = {
                    "file_name": rel_img_str,
                    "text": current_caption,
                    "conditioning_image": rel_cond_str,
                    "faceid_embedding": rel_embed_str,
                    "source_faceid_embedding": source_embed_str,
                    "source_text": source_caption
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
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--captions_json", type=str, required=True, help="Path to JSON file with captions")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--default_caption", type=str, default="high-quality professional photo of a face")
    parser.add_argument("--faceswap", action="store_true", help="Enable FaceSwap tuple output format")
    
    args = parser.parse_args()
    
    process_images_recursive(
        args.input_dir,
        args.output_dir,
        args.captions_json,
        default_caption=args.default_caption,
        max_samples=args.max_samples,
        is_faceswap=args.faceswap
    )

if __name__ == "__main__":
    main()