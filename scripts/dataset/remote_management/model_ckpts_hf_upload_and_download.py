import os
from huggingface_hub import HfApi

HF_TOKEN = ""     
REPO_ID = "" # <username>/<repo_name>
LOCAL_DIR = "" # path to model checkpoints dir
REPO_TYPE = "model"

def upload_checkpoints():
    api = HfApi(token=HF_TOKEN)

    print(f"Ensuring repo '{REPO_ID}' exists")
    api.create_repo(
        repo_id=REPO_ID, 
        repo_type=REPO_TYPE, 
        private=False,
        exist_ok=True
    )

    print(f"Uploading {LOCAL_DIR} to {REPO_ID}")
    api.upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="Upload checkpoints"
    )
    
    print("Upload complete!")


def download_repo():
    print(f"Downloading from {REPO_ID} to {LOCAL_DIR}")
    
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        token=HF_TOKEN,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,  # False to download actual files instead of shortcuts
        # allow_patterns=["checkpoint-10000/*"], # Download checkpoints-10000 only
    )
    
    print("Download complete!")


if __name__ == "__main__":
    if UPLOAD:
        upload_checkpoints()
    else:
        download_repo()