import os
import random
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def prepare_test_set(precomputed_dir, output_dir, num_samples=1000):
    """
    Prepares a test set using ONLY the metadata.jsonl found in precomputed_dir.
    It strictly filters for folders Part1 through Part7.
    """
    root_path = Path(precomputed_dir)
    out_path = Path(output_dir)
    metadata_path = root_path / "metadata.jsonl"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata.jsonl at {metadata_path}")

    # Define output subdirectories
    source_dir = out_path / "sources"
    target_dir = out_path / "targets"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Filter Metadata
    print(f"Loading metadata from {metadata_path}...")
    valid_entries = []
    
    # STRICT SET: Only allow exactly these folder names
    allowed_folders = set(f"Part{i}" for i in range(1, 8))  # {'Part1', 'Part2', ... 'Part7'}

    with open(metadata_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                fname = entry.get("file_name", "")
                
                # --- FIX: Strict Folder Checking ---
                # Split "Part10/05000.png" -> ["Part10", "05000.png"]
                # This ensures Part10 is NOT matched by Part1 logic
                path_parts = fname.split('/')
                
                if len(path_parts) > 1 and path_parts[0] in allowed_folders:
                    # Ensure the referenced file actually exists
                    img_path = root_path / fname
                    if img_path.exists():
                        valid_entries.append(entry)
                        
            except json.JSONDecodeError:
                continue

    print(f"Found {len(valid_entries)} valid entries in Part1-Part7.")
    
    if len(valid_entries) < num_samples * 2:
        print(f"Warning: Requested {num_samples} pairs but only have {len(valid_entries)} images.")
        print("Reducing sample size to max available.")
        num_samples = len(valid_entries) // 2

    # 2. Shuffle and Split
    random.seed(42)
    random.shuffle(valid_entries)
    
    # Split into two disjoint sets: Sources and Targets
    source_entries = valid_entries[:num_samples]
    target_entries = valid_entries[num_samples:num_samples*2]

    # Data structures for output
    pairs_mapping = {}
    retrieval_db = {}
    inference_list = []
    
    print(f"Generating {num_samples} pairs...")
    
    stats = {"landmarks_copied": 0, "embeddings_copied": 0}

    for src_entry, tgt_entry in tqdm(zip(source_entries, target_entries), total=num_samples):
        
        # --- A. Process Source ---
        src_rel_path = src_entry["file_name"]         # e.g., "Part1/00000.png"
        src_embed_rel = src_entry["faceid_embedding"] # e.g., "embeddings/Part1/00000.pt"
        
        # Flatten Filename: Part1_00000.png
        src_clean_name = src_rel_path.replace("/", "_")
        src_stem = Path(src_clean_name).stem
        
        # Copy Source Image
        final_src_path = source_dir / src_clean_name
        shutil.copy2(root_path / src_rel_path, final_src_path)
        
        # Copy Source Embedding (.pt)
        if src_embed_rel and (root_path / src_embed_rel).exists():
            final_embed_path = source_dir / f"{src_stem}.pt"
            shutil.copy2(root_path / src_embed_rel, final_embed_path)
            stats["embeddings_copied"] += 1
        
        # --- B. Process Target ---
        tgt_rel_path = tgt_entry["file_name"]              # e.g., "Part2/12345.png"
        tgt_landmark_rel = tgt_entry["conditioning_image"] # e.g., "landmarks/Part2/12345.png"
        tgt_text = tgt_entry.get("text", "high quality professional photo")
        
        # Flatten Filename
        tgt_clean_name = tgt_rel_path.replace("/", "_")
        tgt_stem = Path(tgt_clean_name).stem
        
        # Copy Target Image
        final_tgt_path = target_dir / tgt_clean_name
        shutil.copy2(root_path / tgt_rel_path, final_tgt_path)
        
        # Copy Target Landmark
        landmark_path_for_config = None
        if tgt_landmark_rel and (root_path / tgt_landmark_rel).exists():
            # Rename for inference script: Part2_12345_landmarks.png
            final_landmark_name = f"{tgt_stem}_landmarks.png"
            final_landmark_path = target_dir / final_landmark_name
            
            shutil.copy2(root_path / tgt_landmark_rel, final_landmark_path)
            landmark_path_for_config = str(final_landmark_path)
            stats["landmarks_copied"] += 1

        # --- C. Build Metadata ---
        pairs_mapping[src_clean_name] = tgt_clean_name
        retrieval_db[src_clean_name] = src_clean_name
        
        item_config = {
            "source_path": str(final_src_path),
            "target_path": str(final_tgt_path),
            "output_name": tgt_clean_name,
            "prompt": tgt_text
        }
        
        if landmark_path_for_config:
            item_config["control_path"] = landmark_path_for_config

        inference_list.append(item_config)

    # --- 3. Save JSON Files ---
    with open(out_path / "pairs.json", "w") as f:
        json.dump(pairs_mapping, f, indent=2)
        
    with open(out_path / "retrieval_db.json", "w") as f:
        json.dump(retrieval_db, f, indent=2)
        
    with open(out_path / "inference_config.json", "w") as f:
        json.dump(inference_list, f, indent=2)

    print(f"\n[Done] Test set prepared at: {output_dir}")
    print(f" - Sources: {len(source_entries)}")
    print(f" - Targets: {len(target_entries)}")
    print(f" - Embeddings Copied: {stats['embeddings_copied']}")
    print(f" - Landmarks Copied: {stats['landmarks_copied']}")
    print(f" - Configs saved: inference_config.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precomputed_dir", type=str, required=True, 
                        help="Root dir containing metadata.jsonl, images/, landmarks/, embeddings/")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save the prepared test set")
    parser.add_argument("--num_samples", type=int, default=1000)
    
    args = parser.parse_args()
    
    prepare_test_set(
        args.precomputed_dir,
        args.output_dir,
        args.num_samples
    )