#!/bin/bash

# cloning repo, building environment, activating & setting up
git clone https://github.com/chengtan9907/OpenSTL
cd OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop

# this resolves some errors I get when running python setup.py develop
pip install -e .

# this is correct version of timm -- built in download is not capatible 
pip install git+https://github.com/huggingface/pytorch-image-models.git

