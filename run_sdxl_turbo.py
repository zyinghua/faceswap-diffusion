from diffusers import AutoPipelineForText2Image
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device) 

image = pipe(
    prompt="full head shot of a person", 
    num_inference_steps=1,
    guidance_scale=0.0
).images[0]

image.save("sdxl_turbo_result.png")