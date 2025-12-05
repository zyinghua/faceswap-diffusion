from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv


# Example format: "username/repo_name"
MODELS = {
    "short":   "zyinghua/canny_face_controlnet_short_prompt_bf16",   
    "medium":  "zyinghua/canny_face_controlnet_medium_prompt_bf16", 
    "generic": "zyinghua/canny_face_controlnet_generic_prompt"  
}

# Where to save them on OSCAR
LOCAL_BASE_DIR = "/users/erluo/scratch/faceswap-diffusion/checkpoints/comparison"
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    for name, repo_id in MODELS.items():
        print(f"--- Downloading {name} model from {repo_id} ---")
        local_dir = os.path.join(LOCAL_BASE_DIR, name)
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=HF_TOKEN,
                ignore_patterns=["*.msgpack", "*.bin"], # Optional: ignore flax weights if only using pt
                local_dir_use_symlinks=False
            )
            print(f"Successfully downloaded to {local_dir}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")

if __name__ == "__main__":
    main()