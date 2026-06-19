#!/bin/bash
# Deploy cleanWMPg1 to server 31310.
# Run from Windows: ssh -p 31310 root@connect.nmb2.seetacloud.com 'bash -s' < deploy_cleanWMPg1.sh
set -e

DEPLOY_ROOT=/root/cleanWMPg1
CONDA_BASE=/opt/miniconda3_r2wmp   # reuse existing conda install
ENV_NAME=cleanwmpg1
PYTHON_VER=3.8.20

echo "==[1/5] Create conda env $ENV_NAME (Python $PYTHON_VER)=="
if [ -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    echo "  env already exists, skipping create"
else
    $CONDA_BASE/bin/conda create -n $ENV_NAME python=$PYTHON_VER -y -q
fi

source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "==[2/5] Install PyTorch (cu113) + Isaac Gym + legged_gym deps=="
pip install -q --upgrade pip
pip install -q torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html || \
    pip install -q torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
        -f https://download.pytorch.org/whl/torch_stable.html

# Isaac Gym needs ninja
pip install -q ninja

# Common deps (from WMP requirements.txt)
pip install -q setuptools==59.5.0 ruamel_yaml==0.17.4 opencv-contrib-python
pip install -q matplotlib tensorboard

echo "==[3/5] Install Isaac Gym (already extracted in /home/WMP/isaacgym)=="
cd /home/WMP/isaacgym/python
pip install -q -e .

echo "==[4/5] rsync cleanWMPg1 source=="
if [ -d "$DEPLOY_ROOT" ]; then
    echo "  $DEPLOY_ROOT already exists; not overwriting"
else
    mkdir -p $DEPLOY_ROOT
fi

# ENV-only check
echo "==[5/5] Verify imports=="
cd $DEPLOY_ROOT 2>/dev/null || cd /home/WMP
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
python -c "import isaacgym; print('isaacgym OK')"
python -c "from rsl_rl.datasets.g1_motion_loader import AMPLoaderG1; print('AMPLoaderG1 OK')"

echo "DONE. To train:"
echo "  conda activate cleanwmpg1"
echo "  export PYTHONPATH=\$DEPLOY_ROOT:\$DEPLOY_ROOT/legged_gym:\$DEPLOY_ROOT/rsl_rl:\$PYTHONPATH"
echo "  export LEGGED_GYM_ROOT_DIR=\$DEPLOY_ROOT/legged_gym"
echo "  cd \$DEPLOY_ROOT && python legged_gym/scripts/train.py --task=g1_amp --headless --sim_device=cuda:0"