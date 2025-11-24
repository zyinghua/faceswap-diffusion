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


def process_images_recursive(input_dir, output_dir, low_threshold=100, high_threshold=200, 
                            canny_subfolder="canny", default_caption="high-quality professional photo of a face"):
    """
    Process 512x512 images and generate Canny edge versions.
    Creates metadata.jsonl in HuggingFace ImageFolder format.
    
    Dataset structure:
    - output_dir/
      - Part1/00000.png (original images)
      - canny/Part1/00000.png (canny images)
      - metadata.jsonl
    
    Args:
        input_dir: Directory with 512x512 images (use resize_images.py first)
        output_dir: Output directory for dataset (will contain original + canny images)
        low_threshold: Canny lower threshold (default: 100)
        high_threshold: Canny upper threshold (default: 200)
        canny_subfolder: Subfolder for canny images (default: "canny")
        default_caption: Default caption for all images
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
    
    canny_detector = CannyDetector()
    metadata_entries = []
    processed_count = 0
    error_count = 0
    
    for img_file in tqdm(image_files, desc="Processing images"):
        try:
            image = Image.open(img_file).convert("RGB")
            rel_path = img_file.relative_to(input_path)
            
            # Copy original image to output_dir (required for ImageFolder format)
            original_output = output_path / rel_path
            original_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_file, original_output)
            
            # Generate and save Canny edge image
            canny_image = make_canny_condition(image, canny_detector, low_threshold, high_threshold)
            canny_output = output_path / canny_subfolder / rel_path
            canny_output.parent.mkdir(parents=True, exist_ok=True)
            canny_image.save(canny_output)
            
            # Create metadata entry (paths relative to output_dir)
            metadata_entries.append({
                "file_name": str(rel_path),  # Original image (HuggingFace creates "image" column from this, always col 0)
                "text": default_caption,  # Caption (should be col 1 for positional fallback)
                "conditioning_image": str(Path(canny_subfolder) / rel_path),  # Canny image (should be col 2 for positional fallback)
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
    print(f"Dataset saved to: {output_path}")
    print(f"  - Original images: {output_path}/")
    print(f"  - Canny images: {output_path}/{canny_subfolder}/")
    print(f"  - Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Canny edge images and create metadata.jsonl for ControlNet training. "
                    "Input images must be 512x512 (use resize_images.py first)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with 512x512 images (can have subdirectories like Part1/, Part2/, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset (will contain original images, canny images, and metadata.jsonl)"
    )
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=100,
        help="Canny lower threshold (default: 100)"
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=200,
        help="Canny upper threshold (default: 200)"
    )
    parser.add_argument(
        "--canny_subfolder",
        type=str,
        default="canny",
        help="Subfolder name for canny images (default: 'canny')"
    )
    parser.add_argument(
        "--default_caption",
        type=str,
        default="high-quality professional photo of a face",
        help="Default caption for all images"
    )
    
    args = parser.parse_args()
    
    process_images_recursive(
        args.input_dir,
        args.output_dir,
        args.low_threshold,
        args.high_threshold,
        args.canny_subfolder,
        args.default_caption
    )


if __name__ == "__main__":
    main()

