import os
import random
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_test_set(dataset_dir, output_dir, num_samples=1000):
    """
    Randomly samples pairs and organizes them for evaluation.
    Structure:
      output_dir/
        sources/
        targets/
        pairs.json  (Mapping for Inference & ID Similarity)
        retrieval_db.json (Database for ID Retrieval)
    """
    dataset_path = Path(dataset_dir)
    out_path = Path(output_dir)
    source_dir = out_path / "sources"
    target_dir = out_path / "targets"
    
    # Create directories
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Gather all valid images
    valid_exts = {'.png', '.jpg', '.jpeg'}
    all_images = [
        f for f in dataset_path.rglob("*") 
        if f.suffix.lower() in valid_exts and "canny" not in str(f) and "landmarks" not in str(f)
    ]
    
    if len(all_images) < num_samples * 2:
        print(f"Warning: Not enough images ({len(all_images)}) for {num_samples} unique pairs. Using what we have.")
        num_samples = len(all_images) // 2

    print(f"Found {len(all_images)} images. Sampling {num_samples} pairs...")
    
    # 2. Shuffle and Select
    random.shuffle(all_images)
    
    # Split into sources and targets (disjoint sets to ensure cross-identity)
    sources = all_images[:num_samples]
    targets = all_images[num_samples:num_samples*2]
    
    pairs_mapping = {}      # Format: {"source_name": "target_name"} (Used for ID Sim)
    retrieval_db = {}       # Format: {"id": "source_path"} (Used for Retrieval)
    inference_list = []     # List for batch inference script
    
    for src, tgt in tqdm(zip(sources, targets), total=num_samples):
        # Define new filenames to avoid collisions if flattened
        # We prefix with original parent folder to keep some identity info if needed
        src_name = f"{src.parent.name}_{src.name}"
        tgt_name = f"{tgt.parent.name}_{tgt.name}"
        
        # Copy files
        shutil.copy2(src, source_dir / src_name)
        shutil.copy2(tgt, target_dir / tgt_name)
        
        # 1. pairs.json (Source -> Swap)
        # Note: We assume the output swap will have the same name as the TARGET
        pairs_mapping[src_name] = tgt_name
        
        # 2. retrieval_db.json
        # Map a unique ID to the source image path relative to source_dir
        retrieval_db[src_name] = src_name
        
        # 3. Inference List (metadata for batch runner)
        inference_list.append({
            "source_path": str(source_dir / src_name),
            "target_path": str(target_dir / tgt_name),
            "output_name": tgt_name # We save swap with target name for easy FID calc
        })

    # Save Metadata
    with open(out_path / "pairs.json", "w") as f:
        json.dump(pairs_mapping, f, indent=2)
        
    with open(out_path / "retrieval_db.json", "w") as f:
        json.dump(retrieval_db, f, indent=2)
        
    with open(out_path / "inference_config.json", "w") as f:
        json.dump(inference_list, f, indent=2)

    print(f"\n[Done] Test set prepared at: {output_dir}")
    print(f" - Sources: {len(sources)}")
    print(f" - Targets: {len(targets)}")
    print(f" - Configs saved: pairs.json, retrieval_db.json, inference_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Root of your raw image dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to create the test set")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of pairs to generate")
    args = parser.parse_args()
    
    prepare_test_set(args.dataset_dir, args.output_dir, args.num_samples)