#!/bin/bash

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH

source /ext3/env.sh

conda update -n base conda -y
conda clean --all --yes
conda install pip -y
conda install ipykernel -y

unset -f which
which conda
which python
python --version
which pip