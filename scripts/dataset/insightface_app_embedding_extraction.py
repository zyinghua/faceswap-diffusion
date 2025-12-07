
"""
Script for extracting AntelopeV2's face embeddings (with detection)
for a face image directory and save them as pt files.

For the use of IP-Adapter.
"""

import cv2
from insightface.app import FaceAnalysis
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import os


def get_face_embedding(app, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    faces = app.get(image)
    return faces


def process_single_image(app, image_path, output_path):
    """Process a single image and save face embedding if detected."""
    faces = get_face_embedding(app, image_path)
    
    if faces is None:
        return False, "Failed to load image"
    
    if len(faces) == 0:
        # Try with smaller detection size (512x512)
        app.det_model.input_size = (512, 512)
        faces = get_face_embedding(app, image_path)
        
        if len(faces) == 0:
            # Try with even smaller detection size (320x320)
            app.det_model.input_size = (320, 320)
            faces = get_face_embedding(app, image_path)
        
        # Reset to default size
        app.det_model.input_size = (640, 640)
        
        if len(faces) == 0:
            return False, "No faces detected"
    
    # Sort to find the largest face (in case background faces are detected)
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    
    # Extract embedding from the largest face
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    # ============== DEBUG VISUALIZATION START =================
    # print(f"faces: {len(faces)}")
    # os.makedirs("debug_check", exist_ok=True)

    # # Draw the bounding box (Green)
    # bbox = faces[0].bbox.astype(int)
    # debug_img = cv2.imread(image_path).copy()
    # cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # if faces[0].kps is not None:
    #     for kp in faces[0].kps:
    #         cv2.circle(debug_img, (int(kp[0]), int(kp[1])), 2, (0, 0, 255), -1)

    # # Save it to check visually
    # cv2.imwrite(f"debug_check/debug_{os.path.basename(image_path)}", debug_img)
    # print(f"face emb size: {faceid_embeds.shape}")
    # ============== DEBUG VISUALIZATION END =================

    # Save embedding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(faceid_embeds, output_path)
    
    return True, None


def process_images_recursive(input_dir, output_dir):
    """
    Process all images in input_dir and save face embeddings to output_dir.
    Preserves folder structure (e.g., Part1/00000.png -> Part1/00000.pt).
    
    Args:
        input_dir: Directory with images (can have subdirectories like Part1/, Part2/, etc.)
        output_dir: Output directory for .pt files (will mirror input structure)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    # Initialize InsightFace
    app = FaceAnalysis(name="antelopev2", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    processed_count = 0
    error_count = 0
    no_face_count = 0
    
    for img_file in tqdm(image_files, desc="Extracting IP-Adapter Frozen Encoder Face Embeddings"):
        try:
            rel_path = img_file.relative_to(input_path)
            
            # Change file extension to .pt and create output path
            rel_path_pt = rel_path.with_suffix('.pt')
            # print(f"rel path pt: {rel_path_pt}")
            output_file = output_path / rel_path_pt
            
            success, error_msg = process_single_image(app, img_file, output_file)
            
            if success:
                processed_count += 1
            elif error_msg == "No faces detected":
                no_face_count += 1
                print(f"No faces detected for {img_file}")
            else:
                error_count += 1
                print(f"Error processing {img_file}: {error_msg}")
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            error_count += 1
            continue
    
    print(f"\nProcessed {processed_count} images successfully")
    if no_face_count > 0:
        print(f"Warning: {no_face_count} images skipped (no faces detected)")
    if error_count > 0:
        print(f"Encountered {error_count} errors")
    print(f"Face embeddings saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with images (can have subdirectories like Part1/, Part2/, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for .pt files (will mirror input folder structure)"
    )
    
    args = parser.parse_args()
    
    process_images_recursive(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
