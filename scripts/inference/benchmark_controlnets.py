import argparse
import json
import os
import torch
import gc
from pathlib import Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from tqdm import tqdm

# CONFIGURATION
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

def load_pipeline(ckpt_path):
    print(f"Loading ControlNet: {ckpt_path}")
    controlnet = ControlNetModel.from_pretrained(ckpt_path, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=DTYPE
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()
    return pipe

def flush():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_json", type=str, required=True, help="Path to DETAILED captions.json")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root of dataset containing 'canny' folder")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    
    # Model Paths
    parser.add_argument("--ckpt_short", type=str, required=True)
    parser.add_argument("--ckpt_medium", type=str, required=True)
    parser.add_argument("--ckpt_generic", type=str, required=True)
    
    args = parser.parse_args()
    
    # 1. Load Master List (The Detailed Captions)
    print(f"Loading captions from {args.captions_json}...")
    with open(args.captions_json, 'r') as f:
        all_captions = json.load(f)
    
    # 2. Filter & Pair Data
    # We only keep items where the Canny image actually exists
    valid_samples = []
    print("Verifying files...")
    
    keys = list(all_captions.keys())
    # Optional: Shuffle to get random samples
    import random
    random.seed(42)
    random.shuffle(keys)

    for rel_path in keys:
        if len(valid_samples) >= args.num_samples: break
        
        # Logic: If caption key is "Part1/00000.png", Canny is at "root/canny/Part1/00000.png"
        canny_path = os.path.join(args.dataset_root, "canny", rel_path)
        
        if os.path.exists(canny_path):
            valid_samples.append({
                "id": rel_path,
                "prompt": all_captions[rel_path], # The Detailed Prompt
                "canny_path": canny_path
            })
            
    print(f"Found {len(valid_samples)} valid samples for benchmarking.")

    # 3. Run Benchmark
    models = [
        ("Short", args.ckpt_short),
        ("Medium", args.ckpt_medium),
        ("Generic", args.ckpt_generic)
    ]
    
    os.makedirs(args.output_dir, exist_ok=True)

    for model_name, ckpt_path in models:
        print(f"\n--- Running {model_name} Model ---")
        pipe = load_pipeline(ckpt_path)
        
        for sample in tqdm(valid_samples):
            # Setup Paths
            file_id = os.path.splitext(sample['id'].replace("/", "_"))[0]
            save_folder = os.path.join(args.output_dir, file_id)
            os.makedirs(save_folder, exist_ok=True)
            
            # Load Canny
            control_image = load_image(sample['canny_path'])
            
            # Inference
            generator = torch.manual_seed(42)
            image = pipe(
                sample['prompt'],
                negative_prompt="noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured",
                num_inference_steps=20,
                generator=generator,
                image=control_image,
                guidance_scale=7.5
            ).images[0]
            
            # Save Image
            image.save(os.path.join(save_folder, f"{model_name}.png"))
            
            # Save Metadata (Once)
            if model_name == "Short":
                control_image.save(os.path.join(save_folder, "canny_condition.png"))
                with open(os.path.join(save_folder, "prompt.txt"), "w") as f:
                    f.write(sample['prompt'])

        # Cleanup to free VRAM for next model
        del pipe
        flush()

if __name__ == "__main__":
    main()