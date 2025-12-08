import os
import torch
import sys
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ================= CONFIGURATION =================
# 1. Paths
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
CHECKPOINT_ROOT = "/users/erluo/scratch/faceswap-diffusion/checkpoints/comparison"
OUTPUT_DIR = "/users/erluo/scratch/convergence_grid"

# 2. Test Inputs
CANNY_IMAGE_PATH = "/users/erluo/scratch/canny_dataset/canny/Part1/00059.png"

# Detailed Prompt used for inference
PROMPT = "A close-up photo of a woman with long, straight blonde hair, wearing small gold earrings, and a neutral expression."
NEGATIVE_PROMPT = "faded, noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"

# 3. Grid Settings
STYLES = ["short", "medium", "detailed","generic"]     # Row Labels
STEPS = [5000, 10000, 15000, 20000]                # Column Labels (Add 20000 if you have it)
NUM_INFERENCE_STEPS = 30
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16
# =================================================

def add_labels_to_grid(grid_image, styles, steps, cell_w, cell_h, margin_left=150, margin_top=80):
    """Adds text labels to the grid image."""
    draw = ImageDraw.Draw(grid_image)
    
    # Try to load a font, otherwise use default
    try:
        # Standard path on many Linux systems (including OSCAR usually)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        font = ImageFont.load_default()

    # Draw Column Headers (Steps)
    for i, step in enumerate(steps):
        text = f"{step//1000}k Steps"
        # Center text roughly
        text_w = 200 # approx
        x = margin_left + i * cell_w + (cell_w - text_w) // 2
        draw.text((x, 20), text, fill="black", font=font)

    # Draw Row Labels (Styles)
    for i, style in enumerate(styles):
        text = style.capitalize()
        y = margin_top + i * cell_h + (cell_h - 40) // 2
        draw.text((10, y), text, fill="black", font=font)
    
    return grid_image

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Control Image
    if not os.path.exists(CANNY_IMAGE_PATH):
        print(f"ERROR: Canny image not found at {CANNY_IMAGE_PATH}")
        return
    control_image = load_image(CANNY_IMAGE_PATH).resize((512, 512))
    control_image.save(os.path.join(OUTPUT_DIR, "reference_canny.png"))
    
    # 2. Initialize Pipeline (Load the first available controlnet just to init)
    print("--- Initializing Pipeline ---")
    dummy_ckpt = os.path.join(CHECKPOINT_ROOT, STYLES[0], f"checkpoint-{STEPS[0]}", "controlnet")
    if not os.path.exists(dummy_ckpt):
        print(f"Warning: Default start checkpoint {dummy_ckpt} not found. Script might fail if no checkpoints exist.")

    controlnet = ControlNetModel.from_pretrained(dummy_ckpt, torch_dtype=DTYPE)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL, controlnet=controlnet, torch_dtype=DTYPE, safety_checker=None
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(DEVICE)
    pipe.enable_model_cpu_offload()

    # Store results for grid
    results_map = {} # results_map[style][step] = PIL.Image

    # 3. Loop
    print(f"\nStarting Grid Search: {len(STYLES)} Styles x {len(STEPS)} Checkpoints")
    
    for style in STYLES:
        results_map[style] = {}
        for step in STEPS:
            # Construct path
            ckpt_path = os.path.join(CHECKPOINT_ROOT, style, f"checkpoint-{step}", "controlnet")
            
            if not os.path.exists(ckpt_path):
                print(f" [SKIP] Missing: {style} at {step} steps")
                # Create a placeholder black image
                results_map[style][step] = Image.new('RGB', (512, 512), (0, 0, 0))
                continue
            
            print(f" [RUN] {style} | {step} steps...")
            
            # Hot-swap ControlNet (Fastest method)
            del pipe.controlnet
            torch.cuda.empty_cache()
            new_controlnet = ControlNetModel.from_pretrained(ckpt_path, torch_dtype=DTYPE).to(DEVICE)
            pipe.controlnet = new_controlnet
            
            # Inference
            generator = torch.Generator(device="cpu").manual_seed(SEED)
            image = pipe(
                PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=control_image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=generator,
                controlnet_conditioning_scale=1.0
            ).images[0]
            
            # Save single image
            filename = f"{style}_{step}.png"
            image.save(os.path.join(OUTPUT_DIR, filename))
            results_map[style][step] = image

    # 4. Create Final Grid Image
    print("\nStitching Grid...")
    cell_w, cell_h = 512, 512
    margin_left = 200
    margin_top = 100
    
    grid_width = margin_left + len(STEPS) * cell_w
    grid_height = margin_top + len(STYLES) * cell_h
    
    grid_image = Image.new('RGB', (grid_width, grid_height), 'white')
    
    # Paste images
    for i, style in enumerate(STYLES):
        for j, step in enumerate(STEPS):
            if step in results_map[style]:
                img = results_map[style][step]
                x = margin_left + j * cell_w
                y = margin_top + i * cell_h
                grid_image.paste(img, (x, y))

    # Add text labels
    grid_image = add_labels_to_grid(grid_image, STYLES, STEPS, cell_w, cell_h, margin_left, margin_top)
    
    # Save Grid
    save_path = os.path.join(OUTPUT_DIR, "convergence_grid.png")
    grid_image.save(save_path)
    print(f"DONE! Grid saved to: {save_path}")

if __name__ == "__main__":
    main()