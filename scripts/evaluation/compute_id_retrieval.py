import json
import argparse
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, List, Tuple
from insightface.app import FaceAnalysis

face_app = None

def init_arcface(ctx_id: int = 0, det_size=(640, 640), det_thresh=0.1, model_name='antelopev2', model_root=None):
    """Initialize ArcFace model with configurable settings."""
    global face_app
    if face_app is None:
        print(f"Loading {model_name}...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        root_dir = os.path.abspath(model_root) if model_root else os.path.expanduser('~/.insightface')

        try:
            face_app = FaceAnalysis(name=model_name, root=root_dir, providers=providers)
            face_app.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as e:
            print(f"Warning: Failed to load {model_name}. Fallback to buffalo_l. Error: {e}")
            face_app = FaceAnalysis(name='buffalo_l', providers=providers)
            face_app.prepare(ctx_id=ctx_id, det_size=det_size)

        if hasattr(face_app, 'det_model'):
            face_app.det_model.det_thresh = det_thresh

def compute_embedding(image_path: str) -> np.ndarray:
    """Computes embedding with Dynamic Resizing + Padding Fallbacks."""
    global face_app
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2 failed to read: {image_path}")
  
    # 1. Standard Detection
    faces = face_app.get(img)
    
    # 2. Resize Fallback (Colab Trick)
    if len(faces) == 0:
        original_size = face_app.det_model.input_size
        face_app.det_model.input_size = (512, 512)
        faces = face_app.get(img)
        face_app.det_model.input_size = original_size

    # 3. Padding Fallback
    if len(faces) == 0:
        pad = 200
        img_padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        faces = face_app.get(img_padded)

    if len(faces) == 0:
        raise ValueError(f"No face detected in {os.path.basename(image_path)}")
    
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].embedding

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return np.dot(emb1, emb2)

def find_top_k_matches(query_emb: np.ndarray, embedding_db: Dict[str, np.ndarray], k: int = 5) -> List[str]:
    similarities = {}
    for key, emb in embedding_db.items():
        similarities[key] = cosine_similarity(query_emb, emb)
    
    sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [key for key, _ in sorted_matches[:k]]

def calculate_retrieval_accuracy(source_to_swapped_path: str,
                                 vector_to_filename_path: str, # NEW ARGUMENT
                                 swapped_dir: str = "",
                                 ctx_id: int = 0,
                                 det_size=(640, 640),
                                 det_thresh=0.1,
                                 model_root=None) -> Tuple[float, float]:

    # Initialize
    init_arcface(ctx_id, det_size, det_thresh, model_root=model_root)
    
    if not os.path.isdir(swapped_dir):
        raise ValueError(f"Swapped directory does not exist: {swapped_dir}")

    # Load Mappings
    with open(source_to_swapped_path, 'r') as f:
        source_to_swapped = json.load(f) # pairs.json
    
    with open(vector_to_filename_path, 'r') as f:
        vector_to_filename = json.load(f) # vector_map.json

    # --- 1. Build Source Database (FROM FILE, NOT IMAGES) ---
    print("\nBuilding Source Embedding Database from file...")
    embedding_db = {}
    
    for vec_str, filename in vector_to_filename.items():
        # Convert string list back to numpy array
        vec_list = json.loads(vec_str)
        embedding_db[filename] = np.array(vec_list, dtype=np.float32)
        
    print(f"Database built: {len(embedding_db)} entries loaded.")

    # --- 2. Evaluate Swaps ---
    print("\nEvaluating Swapped Images...")
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    for i, (source_filename, swapped_filename) in enumerate(source_to_swapped.items(), 1):
        
        # Ensure we have the source vector
        if source_filename not in embedding_db:
            # print(f"Skipping {source_filename}: No source vector found.")
            continue
            
        full_swap_path = str(Path(swapped_dir) / swapped_filename)
        
        try:
            # Compute embedding for SWAP (This still needs detection)
            swapped_emb = compute_embedding(full_swap_path)
        except Exception as e:
            # print(f"Failed swap {swapped_filename}: {e}")
            continue
        
        # Search Database
        top5_matches = find_top_k_matches(swapped_emb, embedding_db, k=5)
        
        ground_truth = source_filename
        total_samples += 1
        
        if len(top5_matches) > 0:
            if top5_matches[0] == ground_truth:
                top1_correct += 1
            if ground_truth in top5_matches:
                top5_correct += 1
        
        if i % 100 == 0:
            print(f"Processed {i} pairs...")

    top1_accuracy = top1_correct / total_samples if total_samples > 0 else 0
    top5_accuracy = top5_correct / total_samples if total_samples > 0 else 0
    
    return top1_accuracy, top5_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pairs_json', help="pairs.json")
    parser.add_argument('vector_json', help="The vector_to_filename.json generated by compute_id_similarity.py")
    parser.add_argument('--swapped-dir', required=True, help="Directory with swapped images")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument("--det-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--model-root", type=str, default=None)
    
    args = parser.parse_args()
    
    t1, t5 = calculate_retrieval_accuracy(
        args.pairs_json,
        args.vector_json, # Using vectors instead of raw DB
        swapped_dir=args.swapped_dir,
        ctx_id=args.gpu,
        det_size=tuple(args.det_size),
        det_thresh=args.det_thresh,
        model_root=args.model_root
    )
    
    print("\n" + "="*50)
    print(f"RETRIEVAL RESULTS")
    print("="*50)
    print(f"Top-1 Accuracy: {t1*100:.2f}%")
    print(f"Top-5 Accuracy: {t5*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()