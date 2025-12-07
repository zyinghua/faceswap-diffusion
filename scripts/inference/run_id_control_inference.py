import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import json

# Import the ID Control Pipeline
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, scripts_dir)
from pipelines.pipeline_faceswap import StableDiffusionIDControlPipeline

# Configuration
BASE_MODEL = "Manojb/stable-diffusion-2-1-base"
CONTROLNET_PATH = ""  # Path to trained ControlNet (ends with /controlnet)
IP_ADAPTER_PATH = ""  # Path to IP-Adapter checkpoint (ends with /ip_adapter/ip_adapter.bin)
FACEID_EMBEDDING_PATH = ""  # Path to FaceID embedding .pt file (source ID face to swap in)
CONTROL_IMAGE = ""  # landmarks
MASK_IMAGE = None  # Optional: Path to mask image for inpainting
IMAGE = None  # Optional: Path to source image for inpainting (required when MASK_IMAGE is provided)
METADATA_JSONL_PATH = ""  # Optional: Path to JSONL file containing prompts (each line should have "text" and "file_name")
PROMPT = None  # Text prompt for generation (Override text in metadata if provided)
NUM_INFERENCE_STEPS = 50
OUTPUT_PATH = "./generated_images"  # Path to output dir
SEED = None
DTYPE = torch.float16
NEGATIVE_PROMPT = "noisy, blurry, low contrast, watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
SAMPLE_NUM = 10  # Number of images to generate

# Guidance parameter (default: 7.5):
GUIDANCE_SCALE = 7.5
# controlnet_conditioning_scale: Controls how strongly the model follows the CONTROL IMAGE (default: 1.0)
CONTROLNET_CONDITIONING_SCALE = 1.0
FACEID_EMBEDDING_DIM = 512
# IP-Adapter scale (optional, for controlling strength of face identity)
IP_ADAPTER_SCALE = None # 0.5 - 1.5


def main():
    # Check inputs
    if not CONTROLNET_PATH:
        raise ValueError("CONTROLNET_PATH must be set to the path of the trained ControlNet")
    if not IP_ADAPTER_PATH:
        raise ValueError("IP_ADAPTER_PATH must be set to the path of the IP-Adapter checkpoint")
    if not FACEID_EMBEDDING_PATH:
        raise ValueError("FACEID_EMBEDDING_PATH must be set to the path of the FaceID embedding .pt file")
    if not CONTROL_IMAGE:
        raise ValueError("CONTROL_IMAGE must be set to the path of the control landmark image")
    
    # Validate inpainting inputs
    if MASK_IMAGE:
        if not IMAGE:
            raise ValueError("IMAGE must be provided when MASK_IMAGE is provided for inpainting")
        if not os.path.exists(MASK_IMAGE):
            raise ValueError(f"MASK_IMAGE path does not exist: {MASK_IMAGE}")
        if not os.path.exists(IMAGE):
            raise ValueError(f"IMAGE path does not exist: {IMAGE}")
    
    # Get prompt from metadata or use provided prompt
    prompt = PROMPT
    if not prompt and METADATA_JSONL_PATH:
        control_image_filename = os.path.basename(CONTROL_IMAGE)
        with open(METADATA_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                file_name = entry.get('file_name', '')
                if file_name.endswith(control_image_filename):
                    prompt = entry.get('text', '')
                    break
    
    if not prompt:
        raise ValueError(f"Prompt must be provided either via PROMPT variable or found in {METADATA_JSONL_PATH}")
    
    print(f"Loading ControlNet from: {CONTROLNET_PATH}")
    print(f"Loading IP-Adapter from: {IP_ADAPTER_PATH}")
    print(f"Loading FaceID embedding from: {FACEID_EMBEDDING_PATH}")
    print(f"Control image: {CONTROL_IMAGE}")
    if MASK_IMAGE:
        print(f"Inpainting mode enabled")
        print(f"  Mask image: {MASK_IMAGE}")
        print(f"  Source image: {IMAGE}")
    print(f"Prompt: {prompt}")
    print(f"Dtype: {DTYPE}")
    
    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=DTYPE)
    
    # Load the ID Control Pipeline
    pipe = StableDiffusionIDControlPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        torch_dtype=DTYPE,
        safety_checker=None,  # Disable safety checker for faster inference
    )
    
    # Load IP-Adapter
    pipe.load_ip_adapter_faceid(IP_ADAPTER_PATH, image_emb_dim=FACEID_EMBEDDING_DIM)
    
    # Speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Memory optimization
    pipe.enable_model_cpu_offload()
    
    # Load control image
    control_image = load_image(CONTROL_IMAGE)
    
    # Load mask and source image for inpainting if provided
    mask_image = None
    image = None
    if MASK_IMAGE:
        mask_image = load_image(MASK_IMAGE)
        image = load_image(IMAGE)
    
    # Load FaceID embedding
    faceid_embedding = torch.load(FACEID_EMBEDDING_PATH, map_location="cpu")
    faceid_embedding = faceid_embedding.to(dtype=DTYPE)
    
    # Ensure faceid_embedding has correct shape: [1, embedding_dim] or [embedding_dim]
    # The pipeline will automatically expand it to match batch_size * num_images_per_prompt
    if faceid_embedding.dim() == 1:
        faceid_embedding = faceid_embedding.unsqueeze(0)
    elif faceid_embedding.dim() == 2 and faceid_embedding.shape[0] > 1:
        raise NotImplementedError("Multiple FaceID embeddings are not supported in this cript")
    
    # print(f"FaceID embedding shape: {faceid_embedding.shape}")
    
    # Prepare prompts
    prompts = [prompt] * SAMPLE_NUM
    negative_prompts = [NEGATIVE_PROMPT] * SAMPLE_NUM
    
    # Generate images
    if SEED is not None:
        generator = torch.Generator(device="cpu").manual_seed(SEED)
        print(f"Using seed: {SEED}")
    else:
        generator = None
    
    print(f"Generating {SAMPLE_NUM} image(s)...")
    
    # Prepare pipeline call arguments
    pipe_kwargs = {
        "prompt": prompts,
        "negative_prompt": negative_prompts,
        "control_image": control_image,
        "faceid_embeddings": faceid_embedding,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "generator": generator,
        "guidance_scale": GUIDANCE_SCALE,
        "controlnet_conditioning_scale": CONTROLNET_CONDITIONING_SCALE,
        "ip_adapter_scale": IP_ADAPTER_SCALE,
    }
    
    # Add inpainting parameters if mask_image is provided
    if mask_image is not None:
        pipe_kwargs["mask_image"] = mask_image
        pipe_kwargs["image"] = image
    
    images = pipe(**pipe_kwargs).images
    
    # Save generated images
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for i, image in enumerate(images):
        output_filename = f"faceswap_output_{i}.png"
        output_path = os.path.join(OUTPUT_PATH, output_filename)
        image.save(output_path)
        print(f"Saved image to: {output_path}")
    
    print("Inference complete!")


if __name__ == "__main__":
    main()

