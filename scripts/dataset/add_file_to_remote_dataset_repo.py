from huggingface_hub import HfApi

api = HfApi(token="")

# Upload the file
api.upload_file(
    path_or_fileobj="/root/autodl-tmp/ffhq-dataset512-canny/captions.json", # Your local file path
    path_in_repo="captions.jsonl",                      # Name it will have on the repo
    repo_id="zyinghua/ffhq-dataset512",
    repo_type="dataset",
    commit_message="Add a json file for the dataset captions"
)