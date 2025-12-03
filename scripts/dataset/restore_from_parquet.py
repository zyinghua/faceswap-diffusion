# Restore images from local parquet files

import os
import io
import glob
import argparse
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset_builder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default="/root/autodl-tmp/ff-celeba-hq-dataset512", help="Path to the git cloned folder")
    parser.add_argument("--out_dir", default="/root/autodl-tmp/ff-celeba-hq-dataset512-restored", help="Output folder for images")
    parser.add_argument("--repo_id", default="zyinghua/ff-celeba-hq-dataset512", help="HF Repo ID to fetch label names")
    args = parser.parse_args()

    print(f"Fetching label mapping from {args.repo_id}...")
    try:
        # This downloads only tiny metadata (kbs), not the images.
        builder = load_dataset_builder(args.repo_id)
        label_names = builder.info.features["label"].names
        print(f"Found {len(label_names)} folders (e.g., {label_names[0]}, {label_names[1]}...)")
    except Exception as e:
        print(f"Could not fetch metadata: {e}. Defaulting to generic names.")
        label_names = None

    parquet_files = sorted(glob.glob(os.path.join(args.repo_dir, "**/*.parquet"), recursive=True))
    print(f"Found {len(parquet_files)} parquet files. Starting extraction...")

    global_counter = 0

    for p_file in tqdm(parquet_files):
        # Open file stream
        parquet_ds = pq.ParquetFile(p_file)
        
        # Process in batches of 1000 to prevent 'Killed'
        for batch in parquet_ds.iter_batches(batch_size=1000):
            df = batch.to_pandas()
            
            for _, row in df.iterrows():
                img_bytes = row["image"]["bytes"]
                img = Image.open(io.BytesIO(img_bytes))
                
                label_idx = row["label"]
                if label_names and label_idx < len(label_names):
                    folder_name = label_names[label_idx]
                else:
                    folder_name = "images" 
                
                save_dir = os.path.join(args.out_dir, folder_name)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{global_counter:05d}.png")
                img.save(save_path)
                
                global_counter += 1

    print(f"Done! Extracted {global_counter} images to {args.out_dir}")

if __name__ == "__main__":
    main()