#!/bin/bash
# Full PPO training step test (10 iterations).
set -e
DEPLOY_ROOT=/root/cleanWMPg1
export PATH=/opt/miniconda3_r2wmp/bin:$PATH
source /opt/miniconda3_r2wmp/etc/profile.d/conda.sh
conda activate cleanwmpg1
export PYTHONPATH=$DEPLOY_ROOT:$DEPLOY_ROOT/legged_gym:$DEPLOY_ROOT/rsl_rl:/home/WMP:$PYTHONPATH
export LEGGED_GYM_ROOT_DIR=$DEPLOY_ROOT/legged_gym
cd $DEPLOY_ROOT

TASK=${1:-g1_amp}
N_ENVS=${2:-16}

echo "=== Full PPO step test: $TASK, $N_ENVS envs ==="

python -u -c "
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch
import sys
TASK = '$TASK'
N_ENVS = $N_ENVS
sys.argv = ['train.py', f'--task={TASK}', '--headless',
            '--sim_device=cuda:0']
args = get_args()
print('args loaded', flush=True)
env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)
env_cfg.env.num_envs = N_ENVS
env_cfg.depth.use_camera = False
train_cfg.runner.num_steps_per_env = 8  # tiny for smoke test
train_cfg.runner.max_iterations = 3
env, env_cfg = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)
print(f'env created; num_obs={env.num_obs}', flush=True)
ppo_runner, train_cfg = task_registry.make_wmp_runner(env=env, name=TASK, args=args, train_cfg=train_cfg)
print('WMPRunner constructed', flush=True)
# Just call learn() for 3 iters (small num_steps_per_env=8 each, total ~96 env steps)
ppo_runner.learn(num_learning_iterations=3, init_at_random_ep_len=True)
print('SUCCESS', flush=True)
" 2>&1 | tail -30