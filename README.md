## Repo for face swapping based on ControlNet and IP-Adapter
Please see train.py for training code

## Environment setup:
Create a conda environment:

``` bash
conda create -n faceswap python=3.10 -y
conda activate faceswap
```

or

``` bash
conda create -p /path/to/conda-env python=3.10 -y
conda activate /path/to/conda-env
```

Then, cd to the repo and install packages:

``` bash
cd faceswap-diffusion/
pip install -r requirements.txt
```

## Download Dataset
Currently, the dataset has been uploaded to [huggingface](https://huggingface.co/datasets/zyinghua/ffhq-dataset512) stored as parquet files.
If your compute has enough bandwidth, you may first try: `scripts/download_and_restore_dataset.py`, which downloads and restore images directly.

**A safer option:**
```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/zyinghua/ffhq-dataset512 # pesudo clone
cd ffhq-dataset512
git config lfs.concurrenttransfers 1 # download only 1 file at a time (saves RAM)
git lfs pull # actual download
python scripts/restore_from_parquet.py # Please change the paths accordingly in the script, or call --repo_dir, etc., params accordingly
```