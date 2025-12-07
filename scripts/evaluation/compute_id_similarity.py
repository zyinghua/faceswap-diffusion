import json
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from pathlib import Path
from typing import Dict, Tuple
import argparse
import onnxruntime as ort


class IDSimilarityCalculator:
    def __init__(self, det_size=(640, 640), use_gpu=True):
        """
        Initialize the ID similarity calculator with ArcFace model.
        
        Args:
            det_size: Detection size for face detection (width, height)
            use_gpu: Whether to use GPU if available (default: True)
        """
        print("Loading ArcFace model...")
        
        # Configure providers based on GPU availability
        if use_gpu:
            try:
                available_providers = ort.get_available_providers()
                
                if 'CUDAExecutionProvider' in available_providers:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    ctx_id = 0  
                else:
                    providers = ['CPUExecutionProvider']
                    ctx_id = -1 
            except:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
        
        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract ArcFace features from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector (512-dimensional embedding)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise ValueError(f"No face detected in image: {image_path}")
        
        face = faces[0]
        embedding = face.normed_embedding
        
        return embedding
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        
        Args:
            feat1: First feature vector
            feat2: Second feature vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Reshape for sklearn's cosine_similarity
        feat1 = feat1.reshape(1, -1)
        feat2 = feat2.reshape(1, -1)
        
        similarity = cosine_similarity(feat1, feat2)[0][0]
        return float(similarity)
    
    def compute_similarities_from_json(
        self, 
        mappings_file: str,
        source_dir: str = "",
        swapped_dir: str = ""
    ) -> Tuple[Dict[str, Tuple[float, str, str]], Dict[str, np.ndarray]]:
        """
        Compute similarities for all image pairs defined in a JSON file.
        
        Args:
            mappings_file: Path to JSON file with mappings
            source_dir: Directory containing source images
            swapped_dir: Directory containing swapped images
            
        Returns:
            Tuple of:
            - Dictionary with source image as key and (similarity, source_path, swapped_path) as value
            - Dictionary mapping source image filenames to their feature vectors
        """
        # Load mappings
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        
        results = {}
        source_vectors = {} 
        source_base = Path(source_dir) if source_dir else Path()
        swapped_base = Path(swapped_dir) if swapped_dir else Path()
        
        for idx, (source_img, swapped_img) in enumerate(mappings.items(), 1):
            source_path = str(source_base / source_img)
            swapped_path = str(swapped_base / swapped_img)
            
            try:
                # Extract features
                source_feat = self.extract_features(source_path)
                swapped_feat = self.extract_features(swapped_path)
                
                # Store source vector
                source_vectors[source_img] = source_feat
                
                # Compute similarity
                similarity = self.compute_similarity(source_feat, swapped_feat)
                
                results[source_img] = (similarity, source_path, swapped_path)
                
                print(f"[{idx}/{len(mappings)}] {source_img} <-> {swapped_img}: {similarity:.4f}")
                
            except Exception as e:
                print(f"[{idx}/{len(mappings)}] Error processing {source_img}: {str(e)}")
                results[source_img] = (None, source_path, swapped_path)
        
        return results, source_vectors
    
    def print_summary(self, results: Dict[str, Tuple[float, str, str]]):
        valid_similarities = [sim for sim, _, _ in results.values() if sim is not None]
        
        if not valid_similarities:
            print("\nNo valid similarities computed!")
            return
        
        print("\nSUMMARY STATISTICS")
        print("="*60)
        print(f"Total pairs processed: {len(results)}")
        print(f"Successful computations: {len(valid_similarities)}")
        print(f"Failed computations: {len(results) - len(valid_similarities)}")
        print(f"\nMean similarity: {np.mean(valid_similarities):.4f}")
        print(f"Std deviation: {np.std(valid_similarities):.4f}")
        print("="*60)
    
    def save_results(
        self, 
        results: Dict[str, Tuple[float, str, str]], 
        source_vectors: Dict[str, np.ndarray],
        output_file: str
    ):
        """
        Save results and vector-to-filename mapping.
        
        Args:
            results: Dictionary of similarity results
            source_vectors: Dictionary of source image vectors
            output_file: Path to save similarity results
        """
        # Save similarity results
        output_data = {
            source: {
                "similarity": sim,
                "source_path": src_path,
                "swapped_path": swap_path
            }
            for source, (sim, src_path, swap_path) in results.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Create mapping from vector to filename
        output_path = Path(output_file)
        mapping_file = str(output_path.parent / f"{output_path.stem}_vector_to_filename.json")
        
        vector_to_filename = {}
        for filename, vec in source_vectors.items():
            # Convert vector to list for JSON serialization
            vec_list = vec.tolist()
            vector_to_filename[str(vec_list)] = filename
        
        with open(mapping_file, 'w') as f:
            json.dump(vector_to_filename, f, indent=2)
        
        print(f"Vector-to-filename mapping saved to: {mapping_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute ID similarity between source and swapped images using ArcFace"
    )
    parser.add_argument(
        "mappings_file",
        help="Path to JSON file containing source-to-swapped image mappings"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing source images"
    )
    parser.add_argument(
        "--swapped-dir",
        required=True,
        help="Directory containing swapped images"
    )
    parser.add_argument(
        "--output",
        default="similarity_results.json",
        help="Output file for results (default: similarity_results.json)"
    )
    parser.add_argument(
        "--det-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Detection size for face detection (width height, default: 640 640)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = IDSimilarityCalculator(
        det_size=tuple(args.det_size),
        use_gpu=not args.cpu
    )
    
    # Compute similarities and extract source vectors
    results, source_vectors = calculator.compute_similarities_from_json(
        args.mappings_file,
        args.source_dir,
        args.swapped_dir
    )
    
    # Print summary
    calculator.print_summary(results)
    
    # Save results and vectors
    calculator.save_results(results, source_vectors, args.output)


if __name__ == "__main__":
    main()