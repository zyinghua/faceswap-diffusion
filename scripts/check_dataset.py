#!/usr/bin/env python
"""
Demo script to load and inspect a ControlNet training dataset.
Shows the same information that train_controlnet.py uses for sanity checking.
"""

import argparse
from pathlib import Path
from datasets import load_dataset
from PIL import Image


def check_dataset(train_data_dir, image_column=None, caption_column=None, conditioning_image_column=None):
    """
    Load and inspect the dataset, showing the same info train_controlnet.py uses.
    """
    print("=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    
    # Load dataset the same way train_controlnet.py does
    dataset = load_dataset(
        train_data_dir,
        cache_dir=None,
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"Dataset splits: {list(dataset.keys())}") # ['train']
    print(f"Number of training examples: {len(dataset['train'])}") # 70000
    
    # Get column names (same as train_controlnet.py line 633)
    column_names = dataset["train"].column_names
    print(f"\nColumn names: {column_names}") # ['image', 'text', 'conditioning_image']
    
    # Determine column names (same logic as train_controlnet.py)
    if image_column is None:
        image_column = column_names[0]
        print(f"\n[Auto-detected] image_column: {image_column} (from column_names[0])")
    else:
        print(f"\n[User-specified] image_column: {image_column}")
        if image_column not in column_names:
            print(f"ERROR: '{image_column}' not found in columns!")
            return
    
    if caption_column is None:
        if len(column_names) > 1:
            caption_column = column_names[1]
            print(f"[Auto-detected] caption_column: {caption_column} (from column_names[1])")
        else:
            print("ERROR: Cannot determine caption_column")
            return
    else:
        print(f"[User-specified] caption_column: {caption_column}")
        if caption_column not in column_names:
            print(f"ERROR: '{caption_column}' not found in columns!")
            return
    
    if conditioning_image_column is None:
        if len(column_names) > 2:
            conditioning_image_column = column_names[2]
            print(f"[Auto-detected] conditioning_image_column: {conditioning_image_column} (from column_names[2])")
        else:
            print("ERROR: Cannot determine conditioning_image_column")
            return
    else:
        print(f"[User-specified] conditioning_image_column: {conditioning_image_column}")
        if conditioning_image_column not in column_names:
            print(f"ERROR: '{conditioning_image_column}' not found in columns!")
            return
    
    # Inspect first example
    print("\n" + "=" * 80)
    print("Inspecting first example:")
    print("=" * 80)
    
    first_example = dataset["train"][0]
    
    print(f"\n1. {image_column}:")
    img = first_example[image_column]
    if isinstance(img, Image.Image):
        print(f"   Type: PIL Image")
        print(f"   Mode: {img.mode}")
        print(f"   Size: {img.size}")
    elif isinstance(img, str):
        print(f"   Type: String (path)")
        print(f"   Path: {img}")
        # Try to load it
        img_path = Path(train_data_dir) / img
        if img_path.exists():
            loaded_img = Image.open(img_path)
            print(f"   Loaded image - Mode: {loaded_img.mode}, Size: {loaded_img.size}")
        else:
            print(f"   WARNING: Path does not exist: {img_path}")
    else:
        print(f"   Type: {type(img)}")
        print(f"   Value: {img}")
    
    print(f"\n2. {caption_column}:")
    caption = first_example[caption_column]
    print(f"   Type: {type(caption)}")
    print(f"   Value: {caption}")
    if isinstance(caption, str):
        print(f"   Length: {len(caption)} chars")
    
    print(f"\n3. {conditioning_image_column}:")
    cond_img = first_example[conditioning_image_column]
    if isinstance(cond_img, Image.Image):
        print(f"   Type: PIL Image")
        print(f"   Mode: {cond_img.mode}")
        print(f"   Size: {cond_img.size}")
    elif isinstance(cond_img, str):
        print(f"   Type: String (path)")
        print(f"   Path: {cond_img}")
        # Try to load it
        cond_img_path = Path(train_data_dir) / cond_img
        if cond_img_path.exists():
            loaded_cond_img = Image.open(cond_img_path).convert("RGB")
            print(f"   Loaded image - Mode: {loaded_cond_img.mode}, Size: {loaded_cond_img.size}, Type: {type(loaded_cond_img)}")
        else:
            print(f"   WARNING: Path does not exist: {cond_img_path}")
    else:
        print(f"   Type: {type(cond_img)}")
        print(f"   Value: {cond_img}")
    
    # Show a few more examples
    # print("\n" + "=" * 80)
    # print("Sample of first 5 examples:")
    # print("=" * 80)
    
    # for i in range(min(5, len(dataset["train"]))):
    #     example = dataset["train"][i]
    #     print(f"\nExample {i}:")
    #     print(f"  {image_column}: {type(example[image_column]).__name__}")
    #     print(f"  {caption_column}: {str(example[caption_column])[:50]}..." if len(str(example[caption_column])) > 50 else f"  {caption_column}: {example[caption_column]}")
    #     print(f"  {conditioning_image_column}: {type(example[conditioning_image_column]).__name__}")

def main():
    parser = argparse.ArgumentParser(
        description="Check ControlNet training dataset format and structure"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Path to training data directory (same as --train_data_dir for train_controlnet.py)"
    )
    # parser.add_argument(
    #     "--image_column",
    #     type=str,
    #     default=None,
    #     help="Image column name (default: auto-detect from column_names[0])"
    # )
    # parser.add_argument(
    #     "--caption_column",
    #     type=str,
    #     default=None,
    #     help="Caption column name (default: auto-detect from column_names[1])"
    # )
    # parser.add_argument(
    #     "--conditioning_image_column",
    #     type=str,
    #     default=None,
    #     help="Conditioning image column name (default: auto-detect from column_names[2])"
    # )
    
    args = parser.parse_args()
    
    check_dataset(args.train_data_dir)


if __name__ == "__main__":
    main()

