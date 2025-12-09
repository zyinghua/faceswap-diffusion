import json
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from pathlib import Path
from typing import Dict, Tuple
import onnxruntime as ort
import os
import argparse
import sys

class IDSimilarityCalculator:
    def __init__(self, det_size=(640, 640), use_gpu=True):
        """
        Initialize the ID similarity calculator with ArcFace model.
        """
        print("Loading ArcFace model...")

        # Configure providers based on GPU availability
        if use_gpu:
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0
                    print("✓ Using GPU (CUDA)")
                else:
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1
                    print("⚠ GPU not available, using CPU")
            except Exception as e:
                print(f"Error checking providers: {e}")
                providers = ['CPUExecutionProvider']
                ctx_id = -1
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1

        # 1. Try antelopev2 (Best quality)
        try:
            # Adjust root if your models are in a custom folder, e.g., root='./models'
            self.app = FaceAnalysis(name='antelopev2', root="/users/erluo/scratch/faceswap-diffusion/scripts",providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            print(f"✓ Model loaded successfully (antelopev2)")
        except Exception:
            print("antelopev2 not found. Trying default model (buffalo_l)...")
            # 2. Fallback to buffalo_l (Default)
            try:
                self.app = FaceAnalysis(name='buffalo_l', providers=providers)
                self.app.prepare(ctx_id=ctx_id, det_size=det_size)
                print(f"✓ Model loaded successfully (buffalo_l)")
            except Exception as e2:
                 print(f"CRITICAL: Failed to load any model. Error: {e2}")
                 sys.exit(1)

    def extract_features(self, image_path: str, debug=False) -> np.ndarray:
        """
        Extract ArcFace features with robust fallbacks for hard-to-detect faces.
        """
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Strategy 1: Standard Detection
        faces = self.app.get(img)

        # Strategy 2: Resize to 512x512 (Helps with scale issues)
        if len(faces) == 0:
            if debug: print(f"  Retrying {os.path.basename(image_path)} at 512x512...")
            original_size = self.app.det_model.input_size
            self.app.det_model.input_size = (512, 512)
            faces = self.app.get(img)
            self.app.det_model.input_size = original_size  # Reset

        # Strategy 3: Padding (Helps with cropped faces/close-ups)
        if len(faces) == 0:
             if debug: print(f"  Retrying {os.path.basename(image_path)} with padding...")
             pad = 200
             img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
             faces = self.app.get(img_padded)

        if len(faces) == 0:
            raise ValueError(f"No face detected in image: {image_path}")

        # Select largest face by area
        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return faces[0].normed_embedding

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        feat1 = feat1.reshape(1, -1)
        feat2 = feat2.reshape(1, -1)
        return float(cosine_similarity(feat1, feat2)[0][0])

    def compute_similarities_from_json(
        self,
        mappings_file: str,
        source_dir: str = "",
        swapped_dir: str = "",
        debug_first_n: int = 3
    ) -> Tuple[Dict[str, Tuple[float, str, str]], Dict[str, np.ndarray]]:
        
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)

        results = {}
        source_vectors = {}
        source_base = Path(source_dir) if source_dir else Path()
        swapped_base = Path(swapped_dir) if swapped_dir else Path()

        print(f"Processing {len(mappings)} pairs...")

        for idx, (source_img, swapped_img) in enumerate(mappings.items(), 1):
            source_path = str(source_base / source_img)
            swapped_path = str(swapped_base / swapped_img)
            debug = idx <= debug_first_n

            try:
                # Extract features
                source_feat = self.extract_features(source_path, debug=debug)
                swapped_feat = self.extract_features(swapped_path, debug=debug)

                # Store source vector (Crucial for Retrieval Step)
                source_vectors[source_img] = source_feat

                # Compute similarity
                similarity = self.compute_similarity(source_feat, swapped_feat)
                results[source_img] = (similarity, source_path, swapped_path)

                if idx % 100 == 0:
                    print(f"[{idx}/{len(mappings)}] Last sim: {similarity:.4f}")

            except Exception as e:
                if debug: print(f"[{idx}] Error: {e}")
                results[source_img] = (None, source_path, swapped_path)

        return results, source_vectors

    def print_summary(self, results):
        valid_similarities = [sim for sim, _, _ in results.values() if sim is not None]
        if not valid_similarities:
            print("\nNo valid similarities computed!")
            return

        print("\nSUMMARY STATISTICS")
        print("="*60)
        print(f"Total pairs: {len(results)}")
        print(f"Success:     {len(valid_similarities)}")
        print(f"Mean Sim:    {np.mean(valid_similarities):.4f}")
        print(f"Std Dev:     {np.std(valid_similarities):.4f}")
        print("="*60)

    def save_results(self, results, source_vectors, output_file):
        # 1. Save Similarity Results
        output_data = {
            src: {"similarity": sim, "source": s_path, "swap": sw_path}
            for src, (sim, s_path, sw_path) in results.items()
        }
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results to: {output_file}")

        # 2. Save Vector Mapping (CRITICAL for Retrieval Script)
        output_path = Path(output_file)
        mapping_file = str(output_path.parent / f"{output_path.stem}_vector_to_filename.json")
        
        vector_to_filename = {}
        for filename, vec in source_vectors.items():
            vector_to_filename[str(vec.tolist())] = filename

        with open(mapping_file, 'w') as f:
            json.dump(vector_to_filename, f, indent=2)
        print(f"Saved vector map to: {mapping_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs_json")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--swapped-dir", required=True)
    parser.add_argument("--output", default="similarity_results.json")
    parser.add_argument("--det-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    
    calculator = IDSimilarityCalculator(det_size=tuple(args.det_size), use_gpu=(args.gpu >= 0))
    
    results, vectors = calculator.compute_similarities_from_json(
        mappings_file=args.pairs_json,
        source_dir=args.source_dir,
        swapped_dir=args.swapped_dir,
        debug_first_n=5
    )
    
    calculator.print_summary(results)
    calculator.save_results(results, vectors, args.output)

if __name__ == "__main__":
    main()