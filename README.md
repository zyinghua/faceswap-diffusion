# Repo for face swapping based on ControlNet and IP-Adapter

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
Currently, the dataset has been uploaded to [huggingface](https://huggingface.co/datasets/zyinghua/ff-celeba-hq-dataset512) stored as parquet files.
If your compute has enough bandwidth, you may first try: `scripts/dataset/remote_management/download_and_restore_dataset.py`, which downloads and restore images directly.

**A safer option:**
```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/zyinghua/ff-celeba-hq-dataset512 # pesudo clone
cd ff-celeba-hq-dataset512
git config lfs.concurrenttransfers 1 # download only 1 file at a time (saves RAM)
git lfs pull # actual download
cd .. #go back into the faceswap-diffusion directory
python scripts/dataset/restore_from_parquet.py # Please change the paths accordingly in the script, or call --repo_dir, etc., params accordingly
rm -rf .git # (Optional) You may want to remove the git copy, which breaks git, but frees up a lot of storage space
```

## Training
Please firstly download the `glint360k_r100.pth` for the Face ID encoder from [here](https://cloud.tsinghua.edu.cn/d/962ccd4b243442a3a144/?p=%2Fcheckpoints%2FDiffSwap&mode=list).

You will need to set up training environment config first, please run `accelerate config`, and follow the below setup as default (change as needed based on your specific environment).
```bash
- This machine
- multi-GPU
- How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
- Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]: no
- Do you wish to optimize your script with torch dynamo?[yes/NO]: no
- Do you want to use FullyShardedDataParallel? [yes/NO]: no
- Do you want to use Megatron-LM ? [yes/NO]: no
- How many GPU(s) should be used for distributed training? [1]: <input your gpu num setting>
- Please select a choice using the arrow or number keys, and selecting with enter: fp16
```

There are two training scripts:
- `train_controlnet.py` for training solely a ControlNet based on your predefined conditions. `train_controlnet.sh` provides exemplar definition of hyper-parameters.
- `train_controlnet_ip-adapter.py` for training ControlNet + IP-Adapter, similar style as InstantID.
- `train.py` for training our face swap model.