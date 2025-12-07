"""
Script for extracting iResNet100 (Glint360K) face embeddings 
for a face image directory and save them as pt files.

For the use of IP-Adapter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from scripts.models.iresnet import iresnet100
from PIL import Image
import torchvision.transforms as T
import argparse
from tqdm import tqdm


def load_model(model_path="glint360k_r100.pth"):
    """Load and initialize the iResNet100 model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = iresnet100(pretrained=False, fp16=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


transform = T.Compose([
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_embedding(model, device, image_path):
    """Extract face embedding from an image."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model(img_tensor)
    
    return F.normalize(emb, p=2.0, dim=1)


def process_single_image(model, device, image_path, output_path):
    """Process a single image and save face embedding."""
    try:
        emb = get_embedding(model, device, image_path)
        
        # Save embedding
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(emb, output_path)
        
        return True, None
    except Exception as e:
        return False, str(e)


def process_images_recursive(input_dir, output_dir, model_path="glint360k_r100.pth"):
    """
    Process all images in input_dir and save face embeddings to output_dir.
    Preserves folder structure (e.g., Part1/00000.png -> Part1/00000.pt).
    
    Args:
        input_dir: Directory with images (can have subdirectories like Part1/, Part2/, etc.)
        output_dir: Output directory for .pt files (will mirror input structure)
        model_path: Path to the iResNet100 model weights file
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
    
    # Initialize model
    print("Loading iResNet100 model...")
    model, device = load_model(model_path)
    print(f"Model loaded on {device}")
    
    processed_count = 0
    error_count = 0
    
    for img_file in tqdm(image_files, desc="Extracting iResNet100 Face Embeddings"):
        try:
            rel_path = img_file.relative_to(input_path)
            
            # Change file extension to .pt and create output path
            rel_path_pt = rel_path.with_suffix('.pt')
            output_file = output_path / rel_path_pt
            
            success, error_msg = process_single_image(model, device, img_file, output_file)
            
            if success:
                processed_count += 1
            else:
                error_count += 1
                print(f"Error processing {img_file}: {error_msg}")
                
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            error_count += 1
            continue
    
    print(f"\nProcessed {processed_count} images successfully")
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
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/glint360k_r100.pth",
        help="Path to the iResNet100 model weights file (default: glint360k_r100.pth)"
    )
    
    args = parser.parse_args()
    
    process_images_recursive(args.input_dir, args.output_dir, args.model_path)


if __name__ == "__main__":
    main()