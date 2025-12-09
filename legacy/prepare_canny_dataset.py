import argparse
import json
import shutil
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)


def make_canny_condition(image, canny_detector, low_threshold=100, high_threshold=200):
    """Convert an image to Canny edge detection format. Assumes input is 512x512."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    canny = canny_detector(gray, low_threshold, high_threshold)
    canny_rgb = np.stack([canny, canny, canny], axis=-1)
    
    return Image.fromarray(canny_rgb)


def process_images_recursive(input_dir, output_dir, captions_json, generic_prompt, low_threshold=100, high_threshold=200, 
                            canny_subfolder="canny", max_samples=None):
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    captions_dict = {}
    
    # --- Caption Loading Logic ---
    if generic_prompt is not None:
        print(f"Using generic prompt for all images: '{generic_prompt}'")
    else:
        # Load captions from JSON if no generic prompt is provided
        if captions_json is None:
            raise ValueError("You must provide either --captions_json OR --generic_prompt.")
            
        captions_file = Path(captions_json)
        if not captions_file.exists():
            raise FileNotFoundError(f"Captions file not found: {captions_json}")
        
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                captions_dict = json.load(f)
            print(f"Loaded {len(captions_dict)} captions from {captions_file}")
        except Exception as e:
            raise ValueError(f"Could not load captions file {captions_json}: {e}")
    
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process")
    
    # --- MAX SAMPLES LOGIC ---
    if max_samples is not None:
        print(f"Running on first {max_samples} images...")
        image_files = image_files[:max_samples]

    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    canny_detector = CannyDetector()
    metadata_entries = []
    processed_count = 0
    error_count = 0
    missing_caption_count = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            image = Image.open(img_file).convert("RGB")
            rel_path = img_file.relative_to(input_path)
            rel_path_str = str(rel_path)
            
            # --- Determine Caption ---
            if generic_prompt is not None:
                caption = generic_prompt
            else:
                # Lookup in dictionary
                # Try direct path first, then filename fallback
                if rel_path_str in captions_dict:
                    caption = captions_dict[rel_path_str]
                elif img_file.name in captions_dict:
                    caption = captions_dict[img_file.name]
                else:
                    missing_caption_count += 1
                    continue
            
            # Copy original image to output_dir
            original_output = output_path / rel_path
            original_output.parent.mkdir(parents=True, exist_ok=True)
            if not original_output.exists():
                shutil.copy2(img_file, original_output)
            
            # Generate and save Canny edge image
            canny_image = make_canny_condition(image, canny_detector, low_threshold, high_threshold)
            canny_output = output_path / canny_subfolder / rel_path
            canny_output.parent.mkdir(parents=True, exist_ok=True)
            canny_image.save(canny_output)
            
            # Create metadata entry
            metadata_entries.append({
                "file_name": rel_path_str,
                "text": caption,
                "conditioning_image": str(Path(canny_subfolder) / rel_path),
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            error_count += 1
            continue
    
    # Save metadata.jsonl
    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nProcessed {processed_count} images")
    if error_count > 0:
        print(f"Encountered {error_count} errors")
    if missing_caption_count > 0:
        print(f"Warning: {missing_caption_count} images skipped (no caption found in captions file)")
    print(f"Dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Group: Either captions_json OR generic_prompt
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--captions_json", type=str, help="Path to JSON file containing captions")
    group.add_argument("--generic_prompt", type=str, help="A single prompt to use for ALL images (ignores json)")
    
    parser.add_argument("--low_threshold", type=int, default=100)
    parser.add_argument("--high_threshold", type=int, default=200)
    parser.add_argument("--canny_subfolder", type=str, default="canny")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of images")
    
    args = parser.parse_args()
    
    process_images_recursive(
        args.input_dir,
        args.output_dir,
        args.captions_json,
        args.generic_prompt,
        args.low_threshold,
        args.high_threshold,
        args.canny_subfolder,
        args.max_samples
    )


if __name__ == "__main__":
    main()