from pipelines.pipeline_stable_diffusion import StableDiffusionPipeline
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
idx = 1
model = ["stabilityai/sd-turbo", "stabilityai/sdxl-turbo"][idx]
pipeline = [StableDiffusionPipeline, StableDiffusionXLPipeline][idx]

pipe = pipeline.from_pretrained(
    model, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to(device)

my_prompts = [
    "full head portrait of a person"
] * 1

images = pipe(
    prompt=my_prompts, 
    num_inference_steps=4, 
    guidance_scale=0.0
).images

for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")