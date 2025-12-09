
import argparse
from cleanfid import fid
import os

def validate_image_directory(path):
    """Validate that the directory exists and contains images."""
    if not os.path.exists(path):
        raise ValueError(f"Directory does not exist: {path}")
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = os.listdir(path)
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not image_files:
        raise ValueError(f"No valid image files found in: {path}")
    
    print(f"Found {len(image_files)} images in {path}")
    return True


def compute_metrics(real_path, fake_path, batch_size=32, device='cuda', num_workers=4):
    """
    Compute FID, CLIP-FID, and KID metrics.
    
    Args:
        real_path: Path to directory containing real/reference images
        fake_path: Path to directory containing generated/fake images
        batch_size: Batch size for processing (default: 32)
        device: Device to use ('cuda' or 'cpu', default: 'cuda')
        num_workers: Number of workers for data loading (default: 4)
    """
    
    
    # Validate directories
    validate_image_directory(real_path)
    validate_image_directory(fake_path)
    
    # Compute FID (Fréchet Inception Distance)
    fid_score = fid.compute_fid(
        real_path, 
        fake_path,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers
    )
    
    # Compute CLIP-FID
    clip_fid_score = fid.compute_fid(
        real_path,
        fake_path,
        mode="clean",
        model_name="clip_vit_b_32",
        batch_size=batch_size,
        device=device,
        num_workers=num_workers
    )
    
    # Compute KID (Kernel Inception Distance)
    kid_score = fid.compute_kid(
        real_path,
        fake_path,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers
    )
    
    # Summary
    print(f"FID:      {fid_score:.4f}")
    print(f"CLIP-FID: {clip_fid_score:.4f}")
    print(f"KID:      {kid_score:.6f}")
    
    return {
        'fid': fid_score,
        'clip_fid': clip_fid_score,
        'kid': kid_score
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute FID, CLIP-FID, and KID metrics for image quality assessment'
    )
    parser.add_argument(
        '--real_path',
        type=str,
        required=True,
        help='Path to directory containing real/reference images'
    )
    parser.add_argument(
        '--fake_path',
        type=str,
        required=True,
        help='Path to directory containing generated/fake images'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation (default: cuda)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for data loading (default: 4)'
    )
    
    args = parser.parse_args()
    
    try:
        metrics = compute_metrics(
            args.real_path,
            args.fake_path,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()