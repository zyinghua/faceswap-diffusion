import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
cache_dir = "/root/autodl-tmp"
#cache_dir = "/oscar/scratch/erluo/model_cache"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    cache_dir=cache_dir,
)
processor = AutoProcessor.from_pretrained(model_name,cache_dir=cache_dir)

def caption_face_images_batch(pil_images: list[Image.Image], style: str = "medium") -> list[str]:
    """
    Generate captions for a batch of face images using Qwen2.5-VL-7B-Instruct.
    
    Args:
        pil_images: List of PIL Images to caption
        style: Caption style - "medium" (5-10 words) or "short" (minimalist)
        
    Returns:
        List of caption strings, one per image
    """
    messages_templates = {
        "medium": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert face captioner. "
                            "Describe the image using a short, natural phrase. "
                            "Focus primarily on the subject's general identity (e.g., man, woman, baby, girl) "
                            "and their hair style, clothing, accessories, or the background environment. "
                            "Keep the caption brief, typically 5-10 words."
                            "\n\nExamples:"
                            "\n- a woman with long dark hair"
                            "\n- a man in a suit and tie"
                            "\n- a baby wearing a knitted bonnet"
                            "\n- a young girl singing into a microphone"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Describe this face in one brief sentence for a training caption."
                        ),
                    },
                ],
            },
        ],
        "short": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a minimalist face captioner. "
                            "Your goal is to describe the subject using the fewest words possible. "
                            "Follow this format generally: '<Identity> [wearing/with] <Most Significant Attribute>'. "
                            "Or '<Identity> <action> <Attribute>', where use a word to describe the action."
                            "\n\nRules:"
                            "\n1. Identity must be generic: use only 'man', 'woman', 'boy', 'girl', 'baby', or 'person'."
                            "\n2. Attribute must be a major clothing item or accessory (e.g., 'wearing a suit', 'with glasses', 'wearing a red shirt')."
                            "\n3. Output nothing else. No full sentences."
                            "\n\nExample outputs:"
                            "\n- a woman with blonde hair"
                            "\n- a man in a suit with a tie"
                            "\n- a girl wearing a hat"
                            "\n- a woman singing into a microphone"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "Provide the minimalist caption."
                        ),
                    },
                ],
            },
        ],
    }
    
    # Select the message template based on style
    messages = messages_templates.get(style, messages_templates["medium"])
    
    # 1. Build chat prompt for each image
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2. Prepare batched inputs (images and text)
    inputs = processor(
        text=[text] * len(pil_images),
        images=pil_images,
        return_tensors="pt",
    ).to(model.device)

    # 3. Generate captions for the batch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    # Only decode newly generated tokens
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    captions = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    return [caption.strip() for caption in captions]


def process_images_recursive(input_dir, output_json, batch_size=4, style="medium"):
    """
    Generate captions in batches, and save to JSON with image filename as key.
    
    Args:
        input_dir: Directory with images (can have subdirectories)
        output_json: Path to output JSON file
        batch_size: Number of images to process in each batch
    """
    input_path = Path(input_dir)
    output_path = Path(output_json)
    
    # Find all image files
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
        image_files.extend(input_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using batch size: {batch_size}")
    print(f"Using caption style: {style}")
    
    if len(image_files) == 0:
        print(f"No images found in {input_dir}")
        return
    
    captions_dict = {}
    processed_count = 0
    error_count = 0
    
    # Process images in batches
    for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        # Load images for this batch
        batch_images = []
        batch_paths = []
        batch_indices = []
        
        for idx, img_file in enumerate(batch_files):
            try:
                image = Image.open(img_file).convert("RGB")
                rel_path = img_file.relative_to(input_path)
                batch_images.append(image)
                batch_paths.append(str(rel_path))
                batch_indices.append(batch_start + idx)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                error_count += 1
                continue
        
        if len(batch_images) == 0:
            continue
        
        # Generate captions for the batch
        try:
            captions = caption_face_images_batch(batch_images, style=style)
            
            # Store captions in dictionary
            for rel_path, caption in zip(batch_paths, captions):
                captions_dict[rel_path] = caption
                processed_count += 1

        except Exception as e:
            print(f"Error processing batch {batch_start}-{batch_end}: {e}")
            error_count += len(batch_images)
            continue
    
    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessed {processed_count} images")
    if error_count > 0:
        print(f"Encountered {error_count} errors")
    print(f"Captions saved to: {output_path}")


def test():
    """
    Test function to verify the captioning pipeline works.
    """
    
    batch_size = 2
    test_images = []
    for i in range(batch_size):
        random_image_array = np.random.randint(0, 256, size=(512, 512, 3), dtype=np.uint8)
        test_image = Image.fromarray(random_image_array)
        test_images.append(test_image)

    print(f"\nGenerating captions for batch of {batch_size} images...")
    
    try:
        captions = caption_face_images_batch(test_images)
        
        print(f"\nTest successful!")
        for i, caption in enumerate(captions):
            print(f"Image {i+1} caption: {caption}")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for face images using Qwen2.5-VL-7B-Instruct. "
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with images (can have subdirectories like Part1/, Part2/, etc.)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON file (e.g., captions.json)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images to process in each batch (default: 4)"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="medium",
        choices=["medium", "short"],
        help="Caption style"
    )
    
    args = parser.parse_args()
    
    process_images_recursive(args.input_dir, args.output_json, args.batch_size, args.style)


if __name__ == "__main__":
    main()
