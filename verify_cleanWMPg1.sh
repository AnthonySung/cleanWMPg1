#!/bin/bash
# Verify cleanWMPg1 deployment on server.
set -e

DEPLOY_ROOT=/root/cleanWMPg1
export PATH=/opt/miniconda3_r2wmp/bin:$PATH
source /opt/miniconda3_r2wmp/etc/profile.d/conda.sh
conda activate cleanwmpg1
# WMP has no setup.py; we use PYTHONPATH directly.
export PYTHONPATH=$DEPLOY_ROOT:$DEPLOY_ROOT/legged_gym:$DEPLOY_ROOT/rsl_rl:/home/WMP:$PYTHONPATH
export LEGGED_GYM_ROOT_DIR=$DEPLOY_ROOT/legged_gym
cd $DEPLOY_ROOT

echo "==[1/5] isaacgym + torch (isaacgym MUST be imported BEFORE torch)=="
python -c "
import isaacgym
import torch
print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} device count={torch.cuda.device_count()}')
print('isaacgym OK')
"

echo "==[2/5] G1 AMP loader=="
python -c "
from rsl_rl.datasets.g1_motion_loader import AMPLoaderG1
import glob
loader = AMPLoaderG1(device='cpu', time_between_frames=1/30,
    motion_files=glob.glob('datasets/g1/*.csv'),
    preload_transitions=500, num_preload_transitions=500)
print(f'G1 AMP loader OK: obs_dim={loader.observation_dim} trajs={len(loader.trajectories_full)} total_seconds={sum(loader.trajectory_lens):.1f}')
"

echo "==[3/5] dreamer configs_g1.yaml=="
python -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('dreamer/configs_g1.yaml').read_text())
print('g1 yaml num_actions:', cfg['defaults']['num_actions'])
print('g1 yaml task:', cfg['defaults']['task'])
print('g1 yaml dyn_deter:', cfg['defaults']['dyn_deter'])
"

echo "==[4/5] G1 config (G1AMPCfg)=="
python -c "
import importlib.util
spec = importlib.util.spec_from_file_location('g1cfg', 'legged_gym/envs/g1/g1/g1_amp_config.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
cfg = m.G1AMPCfg()
print(f'env_name={cfg.env.env_name} num_actions={cfg.env.num_actions} use_amp={cfg.env.use_amp}')
print(f'num_obs={cfg.env.num_observations} num_priv_obs={cfg.env.num_privileged_obs}')
print(f'asset={cfg.asset.file}')
print(f'motion files count={len(cfg.env.amp_motion_files)}')
print(f'runner_class_name={m.G1AMPCfgPPO().runner_class_name}')
print(f'pd hip_pitch={cfg.control.stiffness[\"hip_pitch\"]}/{cfg.control.damping[\"hip_pitch\"]}')
print(f'pd knee={cfg.control.stiffness[\"knee\"]}/{cfg.control.damping[\"knee\"]}')
print(f'default_joint_angles count={len(cfg.init_state.default_joint_angles)}')
"

echo "==[5/5] wmp_runner + amp_ppo use_amp dispatch=="
python -c "
from rsl_rl.runners.wmp_runner import WMPRunner, _ENV_LOADERS
print('WMPRunner OK')
print('_ENV_LOADERS:', _ENV_LOADERS)
from rsl_rl.algorithms import AMPPPO
import inspect
sig = inspect.signature(AMPPPO.__init__)
print('AMPPPO.__init__ has use_amp param?', 'use_amp' in sig.parameters)
"

echo
echo "============================================="
echo "ALL CHECKS PASSED"
echo "============================================="
echo
echo "To train (G1 AMP, ~24GB GPU at 4096 envs):"
echo "  source /opt/miniconda3_r2wmp/etc/profile.d/conda.sh"
echo "  conda activate cleanwmpg1"
echo "  export PYTHONPATH=/root/cleanWMPg1:/root/cleanWMPg1/legged_gym:/root/cleanWMPg1/rsl_rl:/home/WMP:\$PYTHONPATH"
echo "  export LEGGED_GYM_ROOT_DIR=/root/cleanWMPg1/legged_gym"
echo "  cd /root/cleanWMPg1"
echo "  python legged_gym/scripts/train.py --task=g1_amp --headless --sim_device=cuda:0"