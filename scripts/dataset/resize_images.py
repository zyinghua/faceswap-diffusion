import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def resize_image(image, target_size=(512, 512)):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    
    img_array = np.array(image)
    resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_LINEAR)
    resized_image = Image.fromarray(resized)
    
    return resized_image


def process_images_recursive(input_dir, output_dir, target_size=(512, 512)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png'}
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to resize")
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    processed_count = 0
    error_count = 0
    original_size = None
    
    for img_file in tqdm(image_files, desc="Resizing images"):
        try:
            image = Image.open(img_file).convert("RGB")
            
            if original_size is None:
                original_size = image.size
            
            resized_image = resize_image(image, target_size)
            rel_path = img_file.relative_to(input_path)
        
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            resized_image.save(output_file, quality=95)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            error_count += 1
            continue
    
    if processed_count > 0:
        print(f"\nResized {processed_count} images from {original_size} to {target_size}")
    if error_count > 0:
        print(f"Encountered {error_count} errors during processing")
    print(f"Resized images saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize images recursively from 1024x1024 to 512x512 (or custom size)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root directory containing input images (can have subdirectories like Part1/, Part2/, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory to save resized images (mirrors input structure)"
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=512,
        help="Target width for resized images (default: 512)"
    )
    parser.add_argument(
        "--target_height",
        type=int,
        default=512,
        help="Target height for resized images (default: 512)"
    )
    
    args = parser.parse_args()
    
    # Process images recursively
    process_images_recursive(
        args.input_dir,
        args.output_dir,
        (args.target_width, args.target_height)
    )


if __name__ == "__main__":
    main()

