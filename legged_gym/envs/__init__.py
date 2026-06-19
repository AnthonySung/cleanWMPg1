from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg, A1AMPCfgPPO
from legged_gym.envs.g1.g1.g1_amp_config import (
    G1RoughCfg, G1RoughCfgPPO,
    G1AMPCfg, G1AMPCfgPPO,
)
from .base.legged_robot import LeggedRobot

import os

from legged_gym.utils.task_registry import task_registry

# A1 task (no AMP). cleanWMPg1 does not register this — upstream WMP's plain `a1`
# config never worked end-to-end (num_privileged_obs is None in the parent class,
# which makes the noise scale vector and the actor-vs-priv-obs slicing crash at runtime).
# Use `a1_amp` instead and set `cfg.env.use_amp = False` in your own runner if you
# want to skip AMP imitation.
# task_registry.register("a1",      LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("a1_amp",  LeggedRobot, A1AMPCfg(),  A1AMPCfgPPO())

# cleanWMPg1: G1 tasks. g1_amp = with AMP discriminator (main task).
# `g1` (plain PPO) is registered for completeness; same caveats as `a1` apply.
task_registry.register("g1",      LeggedRobot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register("g1_amp",  LeggedRobot, G1AMPCfg(),  G1AMPCfgPPO())