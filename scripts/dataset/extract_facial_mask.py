import argparse
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm


class FacialMaskExtractor:
    """Extracts facial masks from images using MediaPipe Face Mesh."""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh model."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Face mesh landmark indices for face outline (excluding forehead for tighter mask)
        self.FACE_OVAL = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
    
    def extract_mask(self, image):
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract face oval points
        face_points = []
        h, w = image.shape[:2]
        for idx in self.FACE_OVAL:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            face_points.append([x, y])
        
        # Create mask using convex hull
        face_points = np.array(face_points, dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        
        # Apply morphological operations to smooth the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def extract_mask_from_path(self, image_path):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.extract_mask(image)
    
    def save_mask(self, mask, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), mask)
    
    def create_overlay(self, image, mask):
        """
        Create an overlay visualization of the mask on the original image.
        """
        overlay = image.copy()
        
        # Create a colored mask (green with transparency)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green color
        
        # Blend the colored mask with the original image (30% opacity)
        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # Red contour, 2px thick
        
        return overlay


def process_single_image(extractor,image_path, output_path, debug=False):
    """
    Process a single image and extract facial mask.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return False, f"Could not load image from {image_path}"
        
        mask = extractor.extract_mask(image)
        
        if mask is None:
            return False, "No face detected"
        
        extractor.save_mask(mask, output_path)
        
        if debug:
            overlay = extractor.create_overlay(image, mask)
            output_path_obj = Path(output_path)
            debug_path = output_path_obj.parent / f"{output_path_obj.stem}_overlay{output_path_obj.suffix}"
            output_path_obj = Path(debug_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), overlay)
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def process_images_recursive(input_dir, output_dir, preserve_structure=True, debug=False):
    """
    Process all images in input_dir and save facial masks to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    extractor = FacialMaskExtractor()
    
    processed_count = 0
    error_count = 0
    no_face_count = 0
    
    for img_file in tqdm(image_files, desc="Extracting facial masks"):
        try:
            if preserve_structure:
                rel_path = img_file.relative_to(input_path)
                output_file = output_path / rel_path.with_suffix('.png')
            else:
                output_file = output_path / f"{img_file.stem}.png"
            
            success, error_msg = process_single_image(
                extractor, 
                str(img_file), 
                str(output_file),
                debug=debug
            )
            
            if success:
                processed_count += 1
            elif error_msg == "No face detected":
                no_face_count += 1
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
    print(f"Facial masks saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract facial masks from images using MediaPipe Face Mesh"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to a single image file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for mask (required when using --image_path)"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with images (can have subdirectories)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for masks (required when using --input_dir)"
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten output structure (don't preserve subdirectories)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save overlay visualization images showing the mask on the original image (for validation)"
    )
    
    args = parser.parse_args()
    
    # Check inputs
    if args.image_path and args.input_dir:
        parser.error("Cannot use both --image_path and --input_dir")
    
    if args.image_path:
        if args.output_path is None:
            parser.error("--output_path is required when using --image_path")
        
        extractor = FacialMaskExtractor()
        success, error_msg = process_single_image(
            extractor,
            args.image_path,
            args.output_path,
            debug=args.debug
        )
        
        if success:
            print(f"Facial mask saved to {args.output_path}")
            if args.debug:
                output_path_obj = Path(args.output_path)
                overlay_path = output_path_obj.parent / f"{output_path_obj.stem}_overlay{output_path_obj.suffix}"
                print(f"Debug overlay saved to {overlay_path}")
        else:
            print(f"Error: {error_msg}")
            exit(1)
    
    elif args.input_dir:
        if args.output_dir is None:
            parser.error("--output_dir is required when using --input_dir")
        
        process_images_recursive(
            args.input_dir,
            args.output_dir,
            preserve_structure=not args.flatten,
            debug=args.debug
        )
    
    else:
        parser.error("Must provide either --image_path or --input_dir")


if __name__ == "__main__":
    main()
