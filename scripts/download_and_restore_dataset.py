import os
from datasets import load_dataset
from tqdm import tqdm

REPO_ID = "" # specify the huggingface dataset repo id here (e.g., <username>/ffhq-dataset512)
OUTPUT_DIR = "ffhq-dataset512"

print("Connecting to Hugging Face...")
dataset = load_dataset(REPO_ID, split="train", streaming=True)

if "label" in dataset.features:
    int2str = dataset.features["label"].int2str
else:
    # If no labels exist, save everything to one folder
    int2str = lambda x: "images"

print(f"Restoring dataset to: {OUTPUT_DIR}")


global_counter = 0
for example in tqdm(dataset):
    image = example["image"]
    label_idx = example.get("label")
    
    # Determine folder name (e.g., "Part1", "Part2")
    folder_name = int2str(label_idx) if label_idx is not None else "images"
    
    # Create the folder if it doesn't exist
    folder_path = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # restore the original labelling style (e.g., 00005.png)
    filename = f"{global_counter:05d}.png"
    
    save_path = os.path.join(folder_path, filename)
    image.save(save_path)
    
    global_counter += 1

print("Done! All images are downloaded and restored back to the original png format.")