import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
from insightface.app import FaceAnalysis

face_app = None

def init_arcface(ctx_id: int = 0):
    """Initialize ArcFace model."""
    global face_app
    if face_app is None:
        face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))

def compute_embedding(image_path: str) -> np.ndarray:
    """
    Compute ArcFace embedding for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Embedding vector as numpy array (512-dim)
    """
    global face_app
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
  
    faces = face_app.get(img)
    
    if len(faces) == 0:
        raise ValueError(f"No face detected in image: {image_path}")
    
    return faces[0].embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    return np.dot(emb1, emb2)


def find_top_k_matches(query_emb: np.ndarray, 
                       embedding_db: Dict[str, np.ndarray], 
                       k: int = 5) -> List[str]:
    """
    Find top-k nearest neighbors for a query embedding.
    
    Args:
        query_emb: Query embedding vector
        embedding_db: Dictionary mapping embedding keys to embedding vectors
        k: Number of top matches to return
        
    Returns:
        List of top-k matching keys
    """
    similarities = {}
    for key, emb in embedding_db.items():
        similarities[key] = cosine_similarity(query_emb, emb)
    
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [key for key, _ in sorted_matches[:k]]


def calculate_retrieval_accuracy(source_to_swapped_path: str,
                                 embedding_to_source_path: str,
                                 ctx_id: int = 0) -> Tuple[float, float]:
    """
    Calculate top-1 and top-5 retrieval accuracy for face swapping.
    
    Args:
        source_to_swapped_path: Path to JSON mapping source image to swapped image
        embedding_to_source_path: Path to JSON mapping embedding ID to source image
        ctx_id: GPU device ID (0 for first GPU, -1 for CPU)
        
    Returns:
        Tuple of (top1_accuracy, top5_accuracy)
    """

    init_arcface(ctx_id)
    
    with open(source_to_swapped_path, 'r') as f:
        source_to_swapped = json.load(f)
    
    with open(embedding_to_source_path, 'r') as f:
        embedding_to_source = json.load(f)
    
    # Build embedding database
    embedding_db = {}
    failed_sources = []
    for emb_id, source_path in embedding_to_source.items():
        try:
            embedding_db[emb_id] = compute_embedding(source_path)
        except Exception as e:
            print(f"Warning: Failed to compute embedding for {source_path}: {e}")
            failed_sources.append(source_path)
    
    print(f"Successfully computed {len(embedding_db)} embeddings")
    if failed_sources:
        print(f"Failed to compute {len(failed_sources)} embeddings")
    
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    failed_swapped = []
    
    for i, (source_path, swapped_path) in enumerate(source_to_swapped.items()):
        
        if source_path in failed_sources:
            continue
        
        try:
            swapped_emb = compute_embedding(swapped_path)
        except Exception as e:
            print(f"Warning: Failed to compute embedding for swapped image {swapped_path}: {e}")
            failed_swapped.append(swapped_path)
            continue
        
        top5_matches = find_top_k_matches(swapped_emb, embedding_db, k=5)
        
        ground_truth = None
        for emb_id, src_path in embedding_to_source.items():
            if src_path == source_path:
                ground_truth = emb_id
                break
        
        if ground_truth is None:
            print(f"Warning: No ground truth found for source {source_path}")
            continue
        
        total_samples += 1
        
        if len(top5_matches) > 0 and top5_matches[0] == ground_truth:
            top1_correct += 1

        if ground_truth in top5_matches:
            top5_correct += 1
    
    if failed_swapped:
        print(f"Failed to process {len(failed_swapped)} swapped images")
    
    top1_accuracy = top1_correct / total_samples if total_samples > 0 else 0
    top5_accuracy = top5_correct / total_samples if total_samples > 0 else 0
    
    return top1_accuracy, top5_accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Calculate top-1 and top-5 retrieval accuracy for face swapping using ArcFace'
    )
    parser.add_argument(
        'source_to_swapped',
        type=str,
        help='Path to JSON mapping source image to swapped image'
    )
    parser.add_argument(
        'embedding_to_source',
        type=str,
        help='Path to JSON mapping embedding ID to source image'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (0 for first GPU, -1 for CPU, default: 0)'
    )
    
    args = parser.parse_args()
    
    top1_acc, top5_acc = calculate_retrieval_accuracy(
        args.source_to_swapped,
        args.embedding_to_source,
        args.gpu
    )
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc*100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()