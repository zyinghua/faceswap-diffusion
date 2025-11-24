from diffusers import AutoPipelineForText2Image
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

my_prompts = [
    "full head realistic shot of a person"
] * 5

images = pipe(
    prompt=my_prompts, 
    num_inference_steps=4, 
    guidance_scale=0.0
).images

for i, img in enumerate(images):
    img.save(f"batch_result_{i}.png")