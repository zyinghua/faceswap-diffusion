# Upload image dataset to huggingface in the format of parquet files

from datasets import load_dataset

local_folder_path = "ffhq-dataset512" 
REPO_ID = "" # specify the huggingface dataset repo id here (e.g., <username>/ffhq-dataset512)

# Specify your huggingface access token here
MY_TOKEN=""

print("Loading images...")
dataset = load_dataset("imagefolder", data_dir=local_folder_path, split="train")

print(f"Found {len(dataset)} images. Starting conversion and upload...")
dataset.push_to_hub(REPO_ID, max_shard_size="1GB", token=MY_TOKEN)

print("Done!")