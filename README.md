## Repo for face swapping based on ControlNet and IP-Adapter
Please see train.py for training code

## Environment setup:
Create a conda environment:

``` bash
conda create -p /root/autodl-tmp/faceswap/conda-env python=3.10 -y
```

or

``` bash
conda create -n faceswap python=3.10 -y
```

Followed by:

``` bash
conda activate faceswap
```

Then, install packages:

``` bash
pip install -r requirements.txt
```