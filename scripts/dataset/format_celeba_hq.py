# Format celeba-hq extracted images to match ffhq-dataset512 structure
# Input: repo_dir with "female" and "male" subfolders
# Output: out_dir with Part8, Part9, Part10 subfolders (10000 images each)

import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", required=True, help="Path to the celeba-hq extracted folder (with female/ and male/ subfolders)")
    parser.add_argument("--out_dir", required=True, help="Output directory for formatted images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None for random)")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)

    repo_dir = Path(args.repo_dir)
    out_dir = Path(args.out_dir)
    
    female_dir = repo_dir / "female"
    male_dir = repo_dir / "male"
    
    if not female_dir.exists() and not male_dir.exists():
        raise ValueError(f"Neither 'female' nor 'male' folder found in {repo_dir}")
    
    # Collect all image files from both folders
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    all_images = []
    
    if female_dir.exists():
        for ext in image_extensions:
            all_images.extend(female_dir.glob(ext))
            all_images.extend(female_dir.glob(f"**/{ext}"))
    
    if male_dir.exists():
        for ext in image_extensions:
            all_images.extend(male_dir.glob(ext))
            all_images.extend(male_dir.glob(f"**/{ext}"))
    
    random.shuffle(all_images)
    total_images = len(all_images)
    
    print(f"Found {total_images} images total")
    print(f"  - Images will be randomly distributed across parts")
    print(f"  - Will create Part8 with images 70000-79999 (10000 images)")
    print(f"  - Will create Part9 with images 80000-89999 (10000 images)")
    print(f"  - Will create Part10 with images 90000-99999 (10000 images)")
    
    if total_images < 30000:
        print(f"Warning: Only {total_images} images found, but 30000 are needed for 3 parts")
    
    parts_config = [
        {"name": "Part8", "start_idx": 70000},
        {"name": "Part9", "start_idx": 80000},
        {"name": "Part10", "start_idx": 90000},
    ]
    
    image_counter = 0
    
    # Process each part
    for part in parts_config:
        part_name = part["name"]
        start_idx = part["start_idx"]
        part_dir = out_dir / part_name
        part_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {part_name} (images {start_idx:05d} to {start_idx + 9999:05d})...")
        
        part_start_counter = image_counter
        for i in tqdm(range(10000), desc=part_name):
            if image_counter >= total_images:
                print(f"Warning: Ran out of images at {part_name}, image {i}")
                break
            
            src_image = all_images[image_counter]
            
            dest_filename = f"{start_idx + i:05d}.png"
            dest_path = part_dir / dest_filename
            
            # Copy image (convert to PNG if needed)
            try:
                if src_image.suffix.lower() == '.png':
                    shutil.copy2(src_image, dest_path)
                else:
                    from PIL import Image
                    img = Image.open(src_image)
                    img.save(dest_path, 'PNG')
            except Exception as e:
                print(f"Error processing {src_image}: {e}")
                continue
            
            image_counter += 1
        
        images_in_part = image_counter - part_start_counter
        print(f"Completed {part_name}: {images_in_part} images")
    
    print(f"\nDone! Formatted {image_counter} images to {out_dir}")

if __name__ == "__main__":
    main()

