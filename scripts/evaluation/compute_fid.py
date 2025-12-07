import os
import numpy as np
import scipy.linalg
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from google.colab import drive
from torchvision.models import inception_v3

drive.mount('/content/drive')
basepath = r"/content/drive/MyDrive/CSCI2470_Final"
print(f"Basepath: {basepath}")

def load_inception_model(device):
    """Load Inception-v3 model for feature extraction."""
    try:
        model = inception_v3(weights='IMAGENET1K_V1', transform_input=False)
        model.fc = torch.nn.Identity()  # Remove classification layer
        model = model.to(device)
        model.eval()
        print('Model loaded successfully!')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_image(path, size=299):
    """Load and preprocess a single image."""
    try:
        img = Image.open(path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(img)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def load_images_from_dir(image_dir, max_images=None):
    """Load all images from a directory."""
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise ValueError(f"Directory does not exist: {image_dir}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(list(image_dir.glob(f'*{ext}')))
        image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))

    image_paths = sorted(image_paths)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")

    print(f"Found {len(image_paths)} images")
    return image_paths

def extract_features(images, model, device, batch_size=32):
    """Extract Inception features from a batch of images."""
    model.eval()
    all_features = []


    try:
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                features = model(batch)

                # Handle different model outputs
                if isinstance(features, tuple):
                    features = features[0]

                # Flatten if needed
                if len(features.shape) > 2:
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                    features = features.squeeze(-1).squeeze(-1)

                all_features.append(features.cpu())

                if (i // batch_size + 1) % 5 == 0:
                    print(f"  Processed {min(i+batch_size, len(images))}/{len(images)} images")

        result = torch.cat(all_features, dim=0)
        print(f"  Feature extraction complete. Shape: {result.shape}")
        return result

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise

def calculate_statistics(features):
    """Calculate mean and covariance from features."""
    try:
        features = features.numpy().astype(np.float64)

        # Check if we have enough samples for covariance
        if features.shape[0] < 2:
            raise ValueError(f"Need at least 2 images for FID calculation, but only have {features.shape[0]}")

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    except Exception as e:
        print(f"Error calculating statistics: {e}")
        raise

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate FID between two distributions."""
    try:
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        # Add epsilon to diagonal for numerical stability
        offset = np.eye(sigma1.shape[0]) * eps
        sigma1 = sigma1 + offset
        sigma2 = sigma2 + offset

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # If still not finite, use larger epsilon
        if not np.isfinite(covmean).all():
            print("Warning: Adding larger epsilon for numerical stability")
            offset = np.eye(sigma1.shape[0]) * (eps * 1000)
            sigma1 = sigma1 + offset
            sigma2 = sigma2 + offset
            covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                print(f"Warning: Imaginary component {m}, taking real part")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return float(fid)

    except Exception as e:
        print(f"Error calculating FID: {e}")
        print(f"Matrix shapes - sigma1: {sigma1.shape}, sigma2: {sigma2.shape}")
        print(f"Matrix properties - sigma1 finite: {np.isfinite(sigma1).all()}, sigma2 finite: {np.isfinite(sigma2).all()}")
        raise

def compute_faceswap_fid(swapped_dir, reference_stats_path=None,
                         batch_size=32, max_images=None, device='cuda'):
    """
    Compute FID for swapped face images.

    Args:
        swapped_dir: Directory containing swapped face images
        reference_stats_path: (Optional) Path to precomputed reference statistics (.npz file)
                             If None, will use the swapped images themselves as reference
        batch_size: Batch size for processing
        max_images: Maximum number of images to process (None for all)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary containing statistics and FID scores
    """
    print("="*60)
    print("Face Swap FID Calculator")
    print("="*60)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cpu' and torch.cuda.is_available():
        print("Warning: CUDA is available but CPU was selected")

    try:
        # Load model
        print("\n" + "-"*60)
        model = load_inception_model(device)

        # Load swapped images
        print("\n" + "-"*60)
        print(f"Loading swapped images from: {swapped_dir}")
        swapped_paths = load_images_from_dir(swapped_dir, max_images)

        if len(swapped_paths) < 2:
            print(f"\n{'='*60}")
            print(f"ERROR: Need at least 2 images for FID calculation")
            print(f"Found only {len(swapped_paths)} image(s)")
            print(f"{'='*60}")
            return None

        print("Loading images into tensor...")
        swapped_images = []
        for p in swapped_paths:
            img = load_image(p)
            if img is not None:
                swapped_images.append(img)
        swapped_images = torch.stack(swapped_images)
        print(f"Loaded tensor shape: {swapped_images.shape}")

        # Extract features
        print("\n" + "-"*60)
        swapped_features = extract_features(swapped_images, model, device, batch_size)

        # Calculate statistics
        print("\n" + "-"*60)
        mu_swapped, sigma_swapped = calculate_statistics(swapped_features)

        results = {
            'num_images': len(swapped_paths),
            'mu': mu_swapped,
            'sigma': sigma_swapped
        }

        # If reference stats provided, calculate FID
        if reference_stats_path:
            print(f"\nLoading reference statistics from: {reference_stats_path}")
            ref_stats = np.load(reference_stats_path)
            mu_ref = ref_stats['mu']
            sigma_ref = ref_stats['sigma']

            fid_score = calculate_fid(mu_swapped, sigma_swapped, mu_ref, sigma_ref)

            results['fid_score'] = fid_score

            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Number of swapped images: {len(swapped_paths)}")
            print(f"FID Score: {fid_score:.2f}")
        else:
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"Number of swapped images: {len(swapped_paths)}")
            print("\nNo reference statistics provided.")
            print("To compute FID, provide reference_stats_path parameter.")

        print("\n" + "="*60)
        return results

    except Exception as e:
        print(f"\n\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_statistics(swapped_dir, output_path, batch_size=32, max_images=None, device='cuda'):
    """
    Compute and save statistics for a directory of images.
    Useful for creating reference statistics files.

    Args:
        swapped_dir: Directory containing images
        output_path: Path to save statistics (.npz file)
        batch_size: Batch size for processing
        max_images: Maximum number of images to process
        device: Device to use ('cuda' or 'cpu')
    """
    results = compute_faceswap_fid(swapped_dir, None, batch_size, max_images, device)

    if results:
        print(f"\nSaving statistics to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(output_path, mu=results['mu'], sigma=results['sigma'])
        print("Statistics saved successfully")
        return True
    return False

save_statistics(
    swapped_dir='/content/drive/MyDrive/CSCI2470_Final/Target',
    output_path='/content/drive/MyDrive/CSCI2470_Final/Saved_Statistics/ffhq-256x256.npz',
    device='cuda'
)


results = compute_faceswap_fid(
    swapped_dir='/content/drive/MyDrive/CSCI2470_Final/Swapped',
    reference_stats_path='/content/drive/MyDrive/CSCI2470_Final/Saved_Statistics/ffhq-256x256.npz',
    batch_size=32,
    device='cuda'
)