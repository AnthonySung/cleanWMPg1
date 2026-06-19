#!/bin/bash
# Smoke test: build env + step once for each task at small scale.
set -e
DEPLOY_ROOT=/root/cleanWMPg1
export PATH=/opt/miniconda3_r2wmp/bin:$PATH
source /opt/miniconda3_r2wmp/etc/profile.d/conda.sh
conda activate cleanwmpg1
export PYTHONPATH=$DEPLOY_ROOT:$DEPLOY_ROOT/legged_gym:$DEPLOY_ROOT/rsl_rl:/home/WMP:$PYTHONPATH
export LEGGED_GYM_ROOT_DIR=$DEPLOY_ROOT/legged_gym
cd $DEPLOY_ROOT

python <<'PYEOF'
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

# Use upstream-style args (no manual physics_engine set)
import sys
sys.argv = ["smoke.py", "--task=g1_amp", "--headless",
            "--sim_device=cuda:0", "--rl_device=cuda:0",
            "--num_envs=4"]

for task_name in ["a1_amp", "g1_amp", "a1", "g1"]:
    args = get_args()
    args.task = task_name
    args.num_envs = 4
    args.headless = True
    args.sim_device = "cuda:0"
    args.rl_device = "cuda:0"
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = 4
    env_cfg.depth.camera_num_envs = 4
    env, env_cfg = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.reset()
    print(f"[{task_name}] reset OK; obs shape = {obs.shape}")
    actions = torch.zeros(env.num_envs, env.num_actions, device="cuda:0")
    obs, privileged_obs, rewards, dones, infos, reset_ids, terminal_amp = env.step(actions)
    print(f"[{task_name}] step OK; reward mean = {rewards.mean().item():.4f}, done frac = {dones.float().mean().item():.4f}")
    print(f"[{task_name}] obs shape={obs.shape}, priv_obs={privileged_obs.shape if privileged_obs is not None else None}")
    if "depth" in infos:
        print(f"[{task_name}] depth image shape: {infos['depth'].shape if hasattr(infos['depth'], 'shape') else infos['depth']}")
    env.close()
    print(f"[{task_name}] env closed")
    print()
print("ALL ENV SMOKE TESTS PASSED")
PYEOF