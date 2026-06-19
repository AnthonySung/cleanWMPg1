<h1>cleanWMPg1</h1>

Clean G1-humanoid adaptation of [bytedance/WMP](https://github.com/bytedance/WMP) (paper: *World Model-based Perception for Visual Legged Locomotion*, Lai et al. 2024, arXiv 2409.16784).

This repository keeps **WMP's algorithm intact** — same world model, same AMP-style adversarial imitation, same depth-predictor pipeline — and only changes the parts that must change when porting from A1 (12-DoF quadruped) to G1 (29-DoF humanoid):

* URDF / asset config
* AMP motion loader (CSV instead of Laikago JSON)
* Per-joint PD gains, default joint angles, termination conditions
* Dreamer world-model `num_actions` (=29)
* Runner dispatch: AMP on/off driven by `env.cfg.env.use_amp`

## Not included (intentionally)

Compared to the `AnthonySung/WMP-g1` repo, the following author-introduced
modules are **deliberately absent** from cleanWMPg1, see `ANALYSIS.md` for
the rationale:

* `DeepKoopman` (`state_estimator_DK.py`, `actor_critic_DKwmp.py`)
* EMLP-based symmetry augmentation (`amp_ppo_sym.py`, `symm_utils.py`)
* "Double-correction" ymloss

If you want any of those, look at WMP-g1 directly — they are independent
research ideas, not part of WMP the paper.

## Tasks

| Task       | Robot | Use AMP | Description |
|------------|-------|---------|-------------|
| `a1`       | A1    | No      | Plain PPO (carried over from upstream for sanity check) |
| `a1_amp`   | A1    | Yes     | WMP paper baseline on A1 |
| `g1`       | G1    | No      | Vanilla PPO on G1 |
| `g1_amp`   | G1    | Yes     | WMP paper's AMP imitation adapted to G1 (this is the **main task**) |

## File map

```
cleanWMPg1/
├── ANALYSIS.md                         # WMP vs WMP-g1 diff analysis (WMP-g1 audit report: see below)
├── legged_gym/
│   ├── envs/
│   │   ├── __init__.py                 # registers a1, a1_amp, g1, g1_amp
│   │   ├── a1/                         # unchanged
│   │   └── g1/g1/
│   │       └── g1_amp_config.py        # G1RoughCfg, G1AMPCfg + PPO
│   ├── scripts/{train,play}.py         # unchanged (driven by task name)
│   └── resources/robots/g1_description/  # g1_29dof.urdf + meshes (from unitree_rl_gym)
├── rsl_rl/
│   ├── algorithms/amp_ppo.py           # +use_amp flag
│   ├── datasets/g1_motion_loader.py    # CSV, self-implemented quaternion math
│   ├── runners/wmp_runner.py           # +env_name dispatch (a1 vs g1)
│   └── modules/actor_critic_wmp.py     # unchanged (already config-driven)
├── dreamer/
│   ├── configs.yaml                    # A1 (num_actions=12)
│   └── configs_g1.yaml                 # G1 (num_actions=29)
└── datasets/
    ├── mocap_motions/                  # A1 reference motions (JSON)
    └── g1/                             # G1 reference motions (LAFAN1-style CSV)
```

## Installation

Same as upstream WMP (Python 3.8 recommended, Isaac Gym Preview 3, PyTorch 1.10+cu111+):

```bash
# 1. Conda env
conda create -n cleanwmpg1 python=3.8 -y
conda activate cleanwmpg1

# 2. PyTorch + CUDA matching your driver
pip install torch==1.10.0 torchvision==0.11.0 -f https://download.pytorch.org/whl/cu113

# 3. Isaac Gym (download Preview 3 from NVIDIA; install per upstream README)

# 4. This repo (editable so other paths keep working)
cd rsl_rl   && pip install -e . && cd ..
cd legged_gym && pip install -e .
```

## Training

**Important — depth camera**: the G1 task defaults to `depth.use_camera = False`. Isaac Gym Preview 3's depth-camera rendering path segfaults on the autodl-container's driver stack (nvidia driver 580.x / CUDA 13) — verified empirically. The world model (`dreamer/`) is still constructed but the depth camera buffer is all zeros, which is fine for AMP-PPO but means the world model has no real visual signal. If you want depth-camera training, edit `g1_amp_config.py` and set `class depth: use_camera = True` on a host with a working driver.

```bash
cd cleanWMPg1

# G1 AMP imitation (main task, ~24 GB GPU at 4096 envs)
python legged_gym/scripts/train.py --task=g1_amp --headless --sim_device=cuda:0

# G1 plain PPO (no AMP) — not registered by default; use a1_amp with use_amp=False
# python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0

# A1 AMP (regression check vs upstream WMP)
python legged_gym/scripts/train.py --task=a1_amp --headless --sim_device=cuda:0
```

## Visualization

```bash
python legged_gym/scripts/play.py --task=g1_amp --sim_device=cuda:0
```

## Smoke test

```bash
bash smoke_one.sh g1_amp 16 cuda:0   # 16 envs on GPU
bash smoke_one.sh g1_amp 1 cpu       # 1 env on CPU (slow but driver-agnostic)
```

Set `CLEANWMPG1_DEBUG=1` to print URDF / sim-device diagnostics on stderr.

## Verification vs upstream WMP

* A1 path (`a1_amp`) is byte-equivalent to upstream behaviour: same loader class, same yaml, same algorithm class.
* G1 path replaces only the parts listed in the "Not included" / "File map" sections.

## Citation

If you use this, please cite the upstream paper:

```bibtex
@article{lai2024world,
  title={World Model-based Perception for Visual Legged Locomotion},
  author={Lai, Hang and Cao, Jiahang and Xu, Jiafeng and Wu, Hongtao and Lin, Yunfeng and Kong, Tao and Yu, Yong and Zhang, Weinan},
  journal={arXiv preprint arXiv:2409.16784},
  year={2024}
}
```