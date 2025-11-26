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