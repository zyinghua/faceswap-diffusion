import argparse
import json
import shutil
import numpy as np
import cv2
import torch
import facer
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from tqdm import tqdm

class HRNetLandmarkDetector:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading Face Detector (RetinaFace) and Landmark Detector (HRNet) on {device}...")
        
        self.face_detector = facer.face_detector('retinaface/mobilenet', device=self.device)
        self.landmark_detector = facer.face_aligner('farl/wflw/448', device=self.device)

    def __call__(self, image):
        """
        Input: PIL Image
        Output: Tensor of shape [N, 98, 2] (landmarks) or None if no face found
        """
        # FIX: Use torchvision instead of facer.util
        # Facer expects [B, C, H, W] tensor with values 0-255
        img_tensor = TF.to_tensor(image).to(self.device).unsqueeze(0) * 255.0
        
        # Detect Faces
        with torch.inference_mode():
            faces = self.face_detector(img_tensor)
            
            if 'image_ids' not in faces or len(faces['image_ids']) == 0:
                return None
            
            # Detect Landmarks using HRNet
            faces = self.landmark_detector(img_tensor, faces)
            
        # Return landmarks for the first/largest face
        return faces['alignment'][0].cpu().numpy()

# def draw_landmarks(image_size, landmarks):
#     H, W = image_size
#     canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
#     if landmarks is None:
#         return Image.fromarray(canvas)

#     pts = landmarks.astype(np.int32)

#     def draw_curve(indices, color):
#         valid_indices = [i for i in indices if i < len(pts)]
#         if not valid_indices: return
#         curve_pts = pts[valid_indices]
#         cv2.polylines(canvas, [curve_pts], False, color, 2)

#     # WFLW 98-point Mapping
#     draw_curve(range(0, 33), (255, 255, 255)) # Jaw
#     draw_curve(range(33, 42), (255, 255, 0))  # Left Brow
#     draw_curve(range(42, 51), (255, 255, 0))  # Right Brow
#     draw_curve(range(51, 55), (255, 0, 255))  # Nose Bridge
#     draw_curve(range(55, 60), (255, 0, 255))  # Nose Tip
    
#     if 76 < len(pts):
#         cv2.polylines(canvas, [pts[60:68]], True, (0, 255, 0), 2) # Left Eye
#         cv2.polylines(canvas, [pts[68:76]], True, (0, 255, 0), 2) # Right Eye
    
#     if 97 < len(pts):
#         cv2.polylines(canvas, [pts[76:88]], True, (0, 0, 255), 2) # Outer Mouth
#         cv2.polylines(canvas, [pts[88:98]], True, (0, 0, 255), 2) # Inner Mouth

#     return Image.fromarray(canvas)

def draw_landmarks(image_size, landmarks):
    """
    Draws 98-point WFLW landmarks as DOTS (circles) instead of lines.
    """
    H, W = image_size
    # Create black canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    
    if landmarks is None:
        return Image.fromarray(canvas)

    pts = landmarks.astype(np.int32)

    # Helper to draw individual points
    def draw_points(indices, color):
        # indices can be a range or a list
        valid_indices = [i for i in indices if i < len(pts)]
        for i in valid_indices:
            x, y = pts[i]
            # cv2.circle(image, center, radius, color, thickness=-1 means filled)
            cv2.circle(canvas, (x, y), 3, color, -1)

    # --- DRAWING LOGIC (Same semantic colors, but dots) ---

    # 1. Jaw (White)
    draw_points(range(0, 33), (255, 255, 255))
    
    # 2. Brows (Cyan)
    draw_points(range(33, 42), (255, 255, 0)) # Left
    draw_points(range(42, 51), (255, 255, 0)) # Right
    
    # 3. Nose (Magenta)
    draw_points(range(51, 55), (255, 0, 255)) # Bridge
    draw_points(range(55, 60), (255, 0, 255)) # Tip
    
    # 4. Eyes (Green)
    draw_points(range(60, 68), (0, 255, 0))   # Left
    draw_points(range(68, 76), (0, 255, 0))   # Right
    
    # 5. Mouth (Red)
    draw_points(range(76, 88), (0, 0, 255))   # Outer
    draw_points(range(88, 98), (0, 0, 255))   # Inner

    return Image.fromarray(canvas)

def process_images_recursive(input_dir, output_dir, landmark_subfolder="landmarks", 
                            default_caption="high-quality professional photo of a face", max_samples=None):
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images total.")
    
    if max_samples is not None:
        print(f"Running on first {max_samples} images for inspection...")
        image_files = image_files[:max_samples]
    
    detector = HRNetLandmarkDetector()
    
    metadata_entries = []
    processed_count = 0
    error_count = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            image = Image.open(img_file).convert("RGB")
            rel_path = img_file.relative_to(input_path)
            
            # 1. Copy Original
            original_output = output_path / rel_path
            original_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, original_output)
            
            # 2. Inference (HRNet)
            landmarks = detector(image)
            
            if landmarks is None:
                # Skip images where no face was found
                continue

            # 3. Draw Condition Map
            landmark_image = draw_landmarks((image.height, image.width), landmarks)
            
            # 4. Save Condition Map
            landmark_output = output_path / landmark_subfolder / rel_path
            landmark_output.parent.mkdir(parents=True, exist_ok=True)
            landmark_output = landmark_output.with_suffix('.png') 
            landmark_image.save(landmark_output)
            
            # 5. Metadata
            rel_cond_path = str(Path(landmark_subfolder) / rel_path.with_suffix('.png'))
            
            metadata_entries.append({
                "file_name": str(rel_path),
                "text": default_caption,
                "conditioning_image": rel_cond_path,
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            error_count += 1
            continue
    
    # Save metadata
    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\nSaved {processed_count} images to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of images for testing")
    
    args = parser.parse_args()
    
    process_images_recursive(
        args.input_dir,
        args.output_dir,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main()