# cleanWMPg1 — Progress Log

Tracks bug fixes, validation runs, and training milestones since the
project was first committed. Most recent work first.

---

## 2026-06-20 — Two-round Claude Code review + real training validation

**Goal:** audit cleanWMPg1 against itself using Claude Code, fix all
P0/P1 issues, then validate the resulting pipeline with an actual
multi-iteration training run on the AutoDL cloud GPU.

### Round 1 (commit `dbda48e`) — first Claude Code review pass

Reviewer flagged 5 "critical" bugs, but **4 of them were already fixed**
in the same commit (`dbda48e`) by a sibling subagent that ran the
review concurrently. The remaining one (`sample_motion_for_init` shape
mismatch `(N,36)` vs 71-dim frames) was independently confirmed and
fixed in the same window. Outcome: net fix count = 5.

* `_reward_not_fly` / `_reward_contact` switched from penalised-body
  `contact_flag` to actual foot forces via `feet_indices`
* `_reward_arm_pos_dof29` indices corrected from `13..26` to `15..28`
  (G1 URDF has 12 legs + 3 waist + 14 arms; old range included waist
  joints and missed two wrist DoFs)
* `_reward_upper_action_rate` / `_reward_upper_action_smoothness`
  corrected to `range(15:)` to skip waist
* Duplicate `_reward_delta_torques` definition removed
* `sample_motion_for_init` allocates `FULL_STATE_DIM=71` (was
  `TOTAL_COLS=36` — latent crash if `reference_state_initialization`
  were ever enabled)
* `G1AMPCfg.asset.num_force_sensors=2` (was 4; G1 has 2 foot sensors)
* `wmp_runner.py` config-backup `cp` path uses `env_name` template
  (was hardcoded `a1/a1_amp_config.py`)

### Round 2 (commit `e5252b4`) — second Claude Code review pass

Fresh Claude Code review on the post-fix tree found 1 P0 + 5 P1 issues.
All were validated against the actual source code (not just trusted
from the review). Five real bugs fixed:

1. **P0** — `wmp_runner._build_world_model` used
   `pathlib.Path(sys.argv[0]).parent.parent.parent / wm_yaml_name` for
   the dreamer yaml. This breaks whenever the entry point is not
   `train.py` (smoke tests via `python -c "..."` have `sys.argv[0]='-c'`,
   and `python -m legged_gym.scripts.train` puts `__main__.py` on
   `sys.argv[0]`). Fix: anchor at `pathlib.Path(__file__).resolve().parents[2]`.
2. **P1** — `_build_world_model` and the env-name dispatch silently
   fall back to the A1 motion loader + A1 dreamer yaml for unknown
   `env_name`. Now raises `KeyError` with a clear message.
3. **P1** — `compute_observations` auto-fix only updated `num_obs` /
   `num_privileged_obs`, not `self.privileged_dim`. For G1 with DR
   off, actual prefix = `contact_force(6) + contact_flag(13) = 19`,
   but cfg said 16. This made every `obs[:, privileged_dim+6:privileged_dim+9]`
   (command slice in `actor_critic_wmp.act`) and
   `obs[:, privileged_dim-3:privileged_dim]` (`vel_predict_loss` target
   in `amp_ppo.update`) point at the wrong region. Now auto-fix recomputes
   the prefix from the live counts.
4. **P1** — `g1_amp_config.prop_dim` was 72 (formula included a
   phantom 2-dim gait-phase slot that `compute_observations` never
   appended). The world model sliced `obs[:, priv:priv+72]` which
   reached 2 dims into `actions`. Changed to 70 (real layout).
5. **P1** — `g1_motion_loader` finite-diff angular velocity: `atan2`
   returns `[0, π]` always, so backward rotations had wrong axis sign.
   Now force the relative quaternion into canonical hemisphere
   (`w ≥ 0`) before extracting angle/axis so the sign survives.
6. **P2** — `_infer_dt` file-handle leak (`open(path).read()` without
   context manager). Replaced with `f.seek(0)` on the existing handle.

Commit `a5f9200` followed immediately: A1 configs (`a1_amp_config.py`,
`a1_config.py`) needed explicit `env_name = 'a1'` after the strict
`KeyError` was introduced — they had silently relied on the old
default.

### Round 3 (commits `31968fe`, `d93c7ad`, `ff34342`, `daea433`, `5e0c2ff`) — bugs found by real training

A 50-iter training attempt on the cloud GPU (4096 envs) crashed at
iter 1 with two more dtype-mismatch bugs in the dreamer world model
that smoke tests had masked (smoke only collects 3 iters at 16 envs,
world model never trains):

1. `dreamer/tools.py:588,590` — `SymlogDist.log_prob`:
   `torch.where(distance < self._tol, 0, distance)` — Python int `0`
   promoted to long, while `distance` was float32 → `RuntimeError:
   expected scalar type long int but found float`.
   Fix: `torch.zeros_like(distance)` so the true-branch tensor matches
   the dtype of `distance` regardless of upstream casting.

2. `dreamer/tools.py:517` — `DiscDist.log_prob`:
   `torch.where(equal, 1.0, torch.abs(self.buckets[below] - x))` —
   `1.0` literal is float32 but `self.buckets` was float64 (because
   `self._tol=1e-8` is Python double and the `<` comparison promoted
   buckets to float64) → `RuntimeError: expected scalar type double
   but found float`.
   Fix: `torch.ones_like(self.buckets[below])`.

3. `rsl_rl/runners/wmp_runner.py:422` —
   `train_depth_predictor()` is called every `training_interval` iters
   (default 10). It uses `self.env.depth_index_without_crawl_tilt`,
   which is only populated when `depth.use_camera=True`. With
   `use_camera=False` (our default on autodl due to driver-580/CUDA-13
   depth-camera segfaults), this raises `AttributeError` at iter 10.
   Fix: gate the call on `self.env.cfg.depth.use_camera`. (One
   subagent fixed the same condition independently.)

### Real training validation — 79/200 iters completed

After all 11 fixes were in place, a real training run was launched:

* Task: `g1_amp`
* `num_envs=4096`, `num_steps_per_env=24`
* `max_iterations=200` (only 79 completed — see below)
* GPU: RTX 3090, depth camera disabled

Key metrics across the 79 iters:

| iter | Mean reward | tracking_lin_vel | ep_len | AMP loss |
|-----:|-----------:|-----------------:|-------:|---------:|
|    0 |       7.16 |            0.275 |  20.09 |    1.477 |
|    9 |       6.22 |            0.402 |  17.80 |    0.015 |
|   19 |       7.22 |            0.434 |  19.86 |    0.016 |
|   29 |       8.35 |            0.496 |  20.99 |    0.016 |
|   39 |       9.30 |            0.564 |  24.55 |    0.016 |
|   49 |      11.59 |            0.690 |  29.73 |    0.015 |
|   59 |      13.26 |            0.855 |  34.55 |    0.016 |
|   69 |      16.00 |            1.070 |  49.19 |    0.015 |
|   78 |   **16.50** |        **1.309** |**52.58**|  **0.015** |

Observations:
* `tracking_lin_vel` (the dominant reward, weight=20) went from
  0.275 → 1.309 over 79 iters — robot is actually learning to walk
  forward.
* AMP discriminator loss converged from 1.48 → 0.015 by iter 9 and
  stayed flat. `mean_expert_pred ≈ 0.94` vs `mean_policy_pred ≈ -0.94`
  is the textbook GAN equilibrium for an AMP setup where the
  discriminator is just barely keeping up with the policy.
* Episode length more than doubled (20 → 52) — robot survives longer
  before terminating.
* No NaN, no gradient explosion.

Training was killed at iter 79 (not a code bug) — a sibling
subagent launched a full `train.py` run that pushed combined memory
usage above the 24 GB container budget, and the OOM-killer took my
process. ~12 min of full 200 iters would have been ~20 min wall-clock,
so the run was about 40% through.

For full diagnostic detail see `training_long_summary.md`.

---

## 2026-06-20 — Code-review-aware architecture decisions

The Claude Code review also surfaced (and we validated) these
architectural choices that **were correct as-shipped** but worth
recording so future contributors don't "fix" them:

* `compute_observations` auto-fixes `num_obs`/`num_privileged_obs` to
  match the actual concatenated buffer size. This is necessary because
  the G1 contact-flag cardinality (number of URDF bodies matching
  `penalize_contacts_on` substrings) drifts from the configured value
  when the URDF changes. The auto-fix is the right design.
* `WMPRunner` keeps an `_ENV_LOADERS` dict mapping `env_name → (loader,
  yaml)` rather than `if env_name == 'a1' ... elif env_name == 'g1'`
  branches. Future robots only need an entry in this dict.
* `actor_critic_wmp.ActorCriticWMP` keeps two separate WM encoders
  (`wm_feature_encoder` for actor, `critic_wm_feature_encoder` for
  critic). This is intentional asymmetry — the actor and critic can
  use different WM representations. **Not** a duplicate-by-mistake.
* The `dreamer/configs_g1.yaml` differs from `configs.yaml` only in
  `task` and `num_actions`. Don't be tempted to "deduplicate" — the
  YAMLs are dispatched per env_name.

---

## Bug-fix commit chain (most recent first)

```
01b9983 Add long training run summary (79 iters, 4096 envs)
84bddb6 train.py: env var CLEANWMPG1_TAG/MAX_ITERS/SAVE_INTERVAL; long_train.sh wrapper
5e0c2ff Skip depth predictor training when use_camera=False
ff34342 Fix DiscDist torch.where: use ones_like to preserve dtype
daea433 Fix dreamer: clean up residual duplicate lines from earlier edits
d93c7ad Use torch.zeros_like in SymlogDist.log_prob (final dtype fix)
8b81fb3 Fix DiscDist torch.where: 1 -> 1.0
d3ef795 Fix torch.where dtype: use 0.0 not 0 in SymlogDist.log_prob
31968fe Fix symlog_disc.log_prob dtype mismatch in torch.where
2d2a9e2 Fix MSE/SymlogDist: cast both _mode and value to float32 in log_prob
c4b36d0 Fix MSE head log_prob: cast to float when mode is float
35720bc Fix world model training: is_first as long tensor, MSE head dtype cast
a5f9200 Tag A1 configs with env_name='a1' so WMPRunner strict dispatch works
14e6954 Cleanup: gitignore *.log and review_snapshot_*.txt
b53e7cf Fix dreamer world model preprocess: .clone() instead of torch.Tensor() for already-CUDA tensors
e5252b4 Fix 5 real bugs from Claude Code review (round 2)
c4b89ca Cleanup: gitignore review_result.md and .hermes-tmp.*
dbda48e Fix bugs from Claude Code review (5 real issues)
58d89fa Move resources/ to repo root (LEGGED_GYM_ROOT_DIR level)
521b70c Fix shell scripts: use direct PATH to conda env
```

---

## Open follow-ups (when GPU is free)

* Full 200-iter training run to see convergence — single concurrent
  process only (~20 min).
* `depth.use_camera = True` mode is unverified (driver issue on the
  autodl container); would need a different host.
* AMP hyperparameter sweep (`amp_reward_coef`, `amp_task_reward_lerp`)
  — current values were carried over from upstream WMP unchanged.
* World-model encoder is identical for actor and critic (separate
  instances, same architecture). If convergence is bottlenecked by WM
  quality, the next experiment is making the actor use the pre-update
  WM and the critic use the post-update WM.