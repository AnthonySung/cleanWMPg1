#!/bin/bash
# Smoke test using Isaac Gym CPU pipeline (avoids CUDA driver issues).
set -e
DEPLOY_ROOT=/root/cleanWMPg1
export PATH=/opt/miniconda3_r2wmp/envs/cleanwmpg1/bin:/opt/miniconda3_r2wmp/bin:$PATH
export PYTHONPATH=$DEPLOY_ROOT:$DEPLOY_ROOT/legged_gym:$DEPLOY_ROOT/rsl_rl:/home/WMP:$PYTHONPATH
export LEGGED_GYM_ROOT_DIR=$DEPLOY_ROOT/legged_gym
export CLEANWMPG1_DEBUG=1
cd $DEPLOY_ROOT

TASK=${1:-a1_amp}
N_ENVS=${2:-1}
SIM_DEVICE=${3:-cpu}    # try cpu first to bypass GPU driver issue
echo "=== Testing task: $TASK (num_envs=$N_ENVS, sim_device=$SIM_DEVICE) ==="

python -u -c "
import isaacgym
print('[1] isaacgym imported', flush=True)
from legged_gym.envs import *
print('[2] envs imported', flush=True)
from legged_gym.utils import get_args, task_registry
print('[3] task_registry imported', flush=True)
import torch
import sys
TASK = '$TASK'
N_ENVS = $N_ENVS
SIM_DEVICE = '$SIM_DEVICE'
sys.argv = ['smoke.py', f'--task={TASK}', '--headless',
            f'--sim_device={SIM_DEVICE}', f'--rl_device={SIM_DEVICE}']
args = get_args()
print(f'[4] args: {args.task} sim={args.sim_device} rl={args.rl_device}', flush=True)
env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)
print(f'[5] cfg loaded: num_actions={env_cfg.env.num_actions}, use_amp={env_cfg.env.use_amp}', flush=True)
env_cfg.env.num_envs = N_ENVS
env_cfg.depth.camera_num_envs = N_ENVS
# Disable depth camera for CPU smoke test (renders are slow on CPU)
env_cfg.depth.use_camera = False
print('[6] about to make_env', flush=True)
env, env_cfg = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
print(f'[{TASK}] env created; num_dofs={env.num_dofs}, num_obs={env.num_obs}', flush=True)
obs, privileged_obs = env.reset()
print(f'[{TASK}] reset OK; obs shape = {obs.shape}', flush=True)
actions = torch.zeros(env.num_envs, env.num_actions, device=SIM_DEVICE)
print(f'[{TASK}] about to step', flush=True)
obs, privileged_obs, rewards, dones, infos, reset_ids, terminal_amp = env.step(actions)
print(f'[{TASK}] step OK; reward mean = {rewards.mean().item():.4f}', flush=True)
print(f'[{TASK}] SUCCESS', flush=True)
" 2>&1 | tail -50