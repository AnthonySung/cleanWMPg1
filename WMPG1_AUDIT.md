# WMP-g1 Author-Added Algorithm Modules — Audit Report

> **Scope.** This document audits the three modules that `AnthonySung/WMP-g1` adds on top of upstream `bytedance/WMP`. None of these are part of WMP the paper. `cleanWMPg1` deliberately does **not** include them; this report explains why and where each implementation is questionable.
>
> Modules audited:
> 1. **Deep Koopman (DK)** state estimator — `state_estimator_DK.py`, used by `actor_critic_DKwmp.py`, `amp_ppo_DK.py`, `wmp_runner_DK.py`.
> 2. **EMLP / E(2)-equivariant symmetry loss** — `symm_utils.py`, used by `amp_ppo_sym.py`.
> 3. **"Double-correction" ymloss** — referenced in the README (`UPDATE 7.23`) but only a sketch exists in `state_estimator_DK.py`.
>
> Author's own `actor_critic_DKwmp.py` has the comment `# TODO: 上层传参 num_history, prop_dim` at the constructor signature — the author themselves flagged the module as incomplete.

---

## 1. Deep Koopman (`state_estimator_DK.py`)

### 1.1 What the paper / author claims

The idea is to learn a Koopman operator in a learned latent space such that future states can be predicted linearly: `z_{t+1} = K z_t + B a_t`. The encoder `ϕ` maps prop+action history to `z_t`; the decoder `ϕ^{-1}` maps `z_t` back to state; the linear operator `K, B` give next-latent prediction. This is meant to act as a *state compressor* plugged in front of the actor.

### 1.2 What the code does

* Encoder `ϕ(s)`: MLP over `[s, cos(s_joints), sin(s_joints), s_vel², s_ang²]`.
* `propagate`: a *single* `nn.Linear(latent_dim + 3·action_dim, latent_dim, bias=False)` — this is the only "Koopman operator". The state dependence is purely through the **previous** latent; the action enters through concatenated `[a, cos(a), sin(a)]` and is then linearly mixed.
* Loss is the standard DeepKoopman loss:
  ```
  L = α · (L_recon + L_pred) + L_lin + β · L_inf + γ · ||W||₁ + δ · L_metric
  ```
  where `L_lin = ||K·z_t - ϕ(s_{t+1})||²`, `L_inf = max|s - s_pred|`, `L_metric = ||Δz||₁ - ||Δs||₁` (linearity penalty).

### 1.3 Issues found

| # | Severity | Location | Issue |
|---|---|---|---|
| **D1** | **High** | `Process_history_to_traj`, line 113 | Action is extracted from `obs_history` via `next_obs[..., 8 + 2·num_action : 8 + 3·num_action]`. This relies on the *exact* layout of the history buffer. Because upstream WMP's history is built by `runner.learn` as `obs[priv_dim:priv_dim+6] || obs[priv_dim+9:-height]` (with commands removed), the offsets `8` (priv_dim local indices: 0..5 ang/grav, 6..6+n-1 dof_pos, 6+n..6+2n-1 dof_vel) **do not match**. The action slice here implicitly assumes the history contains `actions` at offset `8 + 2·num_action`, but upstream WMP's history **excludes actions**. As a result the "action" fed to DK is in fact a slice of `dof_vel`. So the module either silently learns the wrong thing or crashes depending on the actor-critic wiring. |
| **D2** | **High** | `loss_fn`, line 152 | `lin_loss = || propagate(z_t, a_t) - encoder(s_{t+1}) ||²`. But `encoder` is called with the **next** state's full prop, not the **next** state's encoder output, so this compares against a feature space that has not been linearly propagated. The gradient still flows (correctly) but the conceptual justification "linearity in latent space" is misleading because `encoder` is non-linear. |
| **D3** | **High** | `loss_fn`, `inf_loss` term | `inf_loss = max|s - ŝ_recon| + max|s - ŝ_pred|`. This is unbounded (max over a batch can be 10²+) and is multiplied by `β=1e-5`. The scale is OK numerically but the term does **not** appear in any standard DeepKoopman paper formulation, so it's a free "engineered" knob. |
| **D4** | **Medium** | `loss_fn`, line 166 | `weight_loss` iterates over `self.encoder.encoder` and `self.decoder.decoder` collecting `||W||₁` of each `nn.Linear`. There are no biases (the modules are Linear-ReLU stacks). This is a hand-rolled L1 regularizer; the standard would be `weight_decay` in `optim.Adam`. With `weight_decay_weight = 1e-7` the effect is negligible. |
| **D5** | **Medium** | `encoder.forward`, line 298 | `cos_dof_pos = cos(obs[..., 6 : 6+n])`, `sin_dof_pos = sin(...)`, `square_dof_vel = (obs[..., 6+n : 6+2n])²`, `square_base_ang_vel = (obs[..., :3])²`. The slicing **assumes** a specific layout. But DK is wrapped through `actor_critic_DKwmp.act`, and the `prop_dim` is configurable through `cfg.policy.prop_dim`. If `prop_dim` doesn't line up with the assumed `(base_ang_vel, projected_gravity, dof_pos, dof_vel)` layout, every cosine/sin/square is computed on the wrong slice. There is no assertion guarding this. |
| **D6** | **Low** | `forward`, line 132 | `propagate(nn.Linear(latent + 3·action, latent, bias=False))` produces a *constant* output for `a=0`. The training data is biased towards small actions early on, so this layer may initially learn to ignore `a`. The `Linear` has no bias, no per-action normalisation, no learnable scaling — it does not have enough capacity to model actuation well. |
| **D7** | **Medium** | `state_estimator_DK.py`, imports | The file imports nothing else, but `g1_motion_loader.py` imports `pose3d`, `motion_util`, `transformations` (third-party libs) that are **not** vendored in `WMP-g1`. So `DKRunner` cannot be imported on a fresh machine without `pip install`ing those packages. The README does not mention this. |
| **D8** | **High** | `amp_ppo_DK.py` line 312, 329 | `DK_total_loss.backward()` is called *after* the AC loss's `.backward()` has already populated the buffers (line 339 in WMP-g1). This means the DK backward pass overwrites / mixes with the AC backward gradients for any parameters shared between `actor_critic` and `deep_koopman`. In `actor_critic_DKwmp.py` there is no parameter sharing — DK is a separate module — so this happens to be fine, but it is **fragile** (changing the model wiring would break it). Also, the requires_grad unfreeze/refreeze loop (lines 326-335) is unnecessary if DK has its own optimiser, which it does (`self.DK_optimizer`). The whole freeze dance is dead code. |

### 1.4 Verdict

The DeepKoopman module as implemented in WMP-g1 is a plausible student project but is **not** a faithful re-implementation of any published DeepKoopman paper that we could match to:

* The encoder uses a hand-rolled observation-function expansion (`cos/sin/square`) that is reasonable but not standard.
* The Koopman propagation is a single dense layer, which limits expressivity.
* The `lin_loss` formulation is conceptually off (compares linear-propagated latent against a non-linear encoder of the next state).
* The action extraction has a layout bug (D1) and the layout assumptions are not enforced (D5).
* The freeze/unfreeze loop (D8) is dead code.

**Recommendation:** Do **not** use this in cleanWMPg1. If you want a real DeepKoopman, use a published reference (e.g. `dstoolkit/dko` or the original Koopman VAE paper) and verify against it.

---

## 2. EMLP symmetry loss (`amp_ppo_sym.py`, `symm_utils.py`)

### 2.1 What the author claims

Encode proprioceptive / command / WM-feature tensors as **scalar / vector / regular representations** of the cyclic group `C₂ = {id, swap(left↔right)}`, then enforce that the actor's output is *equivariant*: `π(σ(x)) = σ(π(x))`, where `σ` is the swap. Implemented as an MSE `||π(x) - σ^{-1}(π(σ(x)))||²`.

### 2.2 What the code does

* Builds a representation map `representations = {name: e2cnn FieldType}` for prop slices (base_ang_vel, projected_grav, dof_pos, dof_vel, phase, last_action, …).
* `get_symm_tensor(tensor, G, representations)` applies `σ` to each slice according to its representation.
* Symmetry loss: run the actor on a swapped history, swap the resulting actions back, and compare to the original action mean.
* Coef `sym_coef = 5.0` (a1) / `1.0` (g1_DK).

### 2.3 Issues found

| # | Severity | Location | Issue |
|---|---|---|---|
| **S1** | **High** | `amp_ppo_sym.py` import | Imports `escnn`, which is **not** in the standard `requirements.txt`. `pip install -r requirements.txt` does not install `escnn`. README does not mention this. Fresh installs will fail at `from rsl_rl.modules import ActorCriticWMP` if the sym module is reachable. |
| **S2** | **High** | `symm_utils.py`, `get_symm_tensor` | The function signature is `(tensor, G, representations_dict)`, but inside it indexes `representations_dict[tensor_key]` by *the channel layout of the history buffer*. This is yet another place where the slice layout of `history_dim` is hard-coded. There is no shape assertion. |
| **S3** | **Medium** | symloss is `||μ_batch - σ(μ(σ(x)))||²` with σ being only horizontal reflection | For a humanoid, there are *more* symmetries worth using (yaw is rotation-equivariant, height channel is invariant, arm joint indices should be swapped). The implementation **only** swaps (left↔right) and does it for the leg joints. For G1 the upper-body joints are **not** invariant under left↔right swap because the URDF defines specific left/right joint names with no "swap" semantic. So either (a) the representation map only swaps the legs and ignores arms, or (b) it swaps arm joints by index which is **wrong** for humanoid — left/right arms do different things (e.g. waving vs guarding). |
| **S4** | **Medium** | `amp_ppo_sym.py` line 415 | The symloss uses `actions_symmetry = actor_critic.act(...)` *after* the actor has already been called with the original `aug_obs_batch` at line 195. Both calls mutate `self.actor_critic.action_mean` etc., so the `sym_loss` uses the **second** call's `action_mean`, not the first. The intent seems to be `μ_batch = first_mean` and `actions_symmetry_reversed = σ^{-1}(second_mean)`, but the code reads `mu_batch = self.actor_critic.action_mean` at line 418, which is now the second call's mean. So `sym_loss = ||σ^{-1}(μ_2(x)) - μ_2(x)||`, **not** the intended `||μ_1(x) - σ^{-1}(μ_2(σ(x)))||`. This is a bug. |
| **S5** | **Low** | `sym_coef` | Set to `5.0` for the no-DK path, `1.0` for the DK path, with no justification. For legged locomotion literature, `sym_coef` is typically 0.1–1.0. The 5.0 value will dominate the actor gradient and prevent task reward learning in early training. |

### 2.4 Verdict

The symmetry augmentation is a sound idea (used in many recent papers, e.g. "Learning Symmetric Embeddings" and the `rl-with-invariant-priors` work) but the WMP-g1 implementation has **multiple bugs** (S1 missing dep, S4 wrong action_mean). Even if the bugs were fixed, `sym_coef = 5.0` is too high.

**Recommendation:** Do not use this in cleanWMPg1. If you want a symmetry prior, build it from the standard `e2cnn` template matching the WMP-g1 idea, but verify each tensor slice with shape assertions and use `sym_coef ≤ 1.0`.

---

## 3. "Double-correction" ymloss (mentioned in README)

### 3.1 What the README says

> `## UPDATE 7.23  添加了双修正机制（h_t 和 a_t），实现DK_ymloss，阶段完成性版本。`

That is, "added the double-correction mechanism (h_t and a_t), implemented `DK_ymloss`, completion version of the stage."

### 3.2 What the code does

* **Grepping for `ymloss`, `w_delta_a`, `w_delta_s`, `h_t`, `a_t` in the WMP-g1 tree returns matches.**
* `actor_critic_DKwmp.act` (line 252–254):
  ```python
  propagated_state = self.deep_koopman.state_propagate(DK_embeded_history[..., -dk_latent_dim:], action)
  delta_action = self.w_delta_a * self.deep_koopman.forward_planner(propagated_state)
  modified_action = action + delta_action   # "double correction" on a_t
  ```
* `actor_critic_DKwmp.HistoryEncoder.forward` (line 351–361):
  ```python
  propagated = self.DK.state_propagate(traj['states'][:, 1:, :], traj['actions'])
  propagated_his = torch.cat((propagated, traj['actions']), dim=-1)
  ...
  propagated_loss = self.w_delta_s * propagated_his - x[..., dk_latent_dim + num_action:]  # "double correction" on h_t
  ```
* The literal string `ymloss` and `DK_ymloss` **never appears** in any `.py` file. The README refers to a `DK_ymloss` function that does not exist; what exists is the `w_delta_a` / `w_delta_s` action and state residual connections.

### 3.3 Issues found

| # | Severity | Location | Issue |
|---|---|---|---|
| **Y1** | **Low** | README | `DK_ymloss` is named in the README but no symbol by that name exists in code. The "double-correction" idea is implemented as two scalar weights `w_delta_a`, `w_delta_s`, both default to `0.0`. |
| **Y2** | **High** | `actor_critic_DKwmp.act`, line 252 | `DK_embeded_history[..., -dk_latent_dim:]` assumes the **last** `dk_latent_dim` channels of the DK-encoded history are the latent from the most recent step. This depends on `deep_koopman.history_encode` ordering. If that ordering ever flips (e.g. oldest-first instead of newest-first), `propagated_state` is wrong by an entire history window. |
| **Y3** | **Medium** | `actor_critic_DKwmp.HistoryEncoder.forward`, line 361 | `propagated_loss = self.w_delta_s * propagated_his - x[..., dk_latent_dim + num_action:]`. The intended loss term is never added to `loss_fn` in `state_estimator_DK.DeepKoopman`. So `w_delta_s` is computed but **never contributes to gradient** — it is a dead branch. |
| **Y4** | **Medium** | `actor_critic_DKwmp.act_inference`, line 280 | Uses `self.deep_koopman.propagate(...)`, while `act` uses `self.deep_koopman.state_propagate(...)`. These are **two different methods** of the DeepKoopman module — we verified `state_propagate` exists but `propagate` is the standard "single linear layer" forward of the Koopman operator. The mismatch means train-time and inference-time action refinement are computed by different functions. This is a subtle bug. |
| **Y5** | **Low** | `w_delta_a` default `0.0` | With the default `w_delta_a = 0.0`, `delta_action = 0` and the action refinement has zero effect. Any reported "DK gains" in the WMP-g1 README must therefore come from the DK reconstruction/prediction losses, **not** the double-correction mechanism. The double-correction is effectively off by default. |

### 3.4 Verdict

The "double-correction" mechanism is partially scaffolded: `w_delta_a` *is* wired into the action output (Y2, Y4) but `w_delta_s` is **dead code** (Y3). The naming `DK_ymloss` in the README is misleading because it suggests a loss term, but no such loss exists. The action residual (`w_delta_a * forward_planner(...)`) is the only part that actually contributes, and only when the user manually sets `w_delta_a ≠ 0` (which `G1CfgDKPPO` in `g1_config.py` does not).

**Recommendation:** Do not advertise this as a feature. If you want action refinement, lift `w_delta_a` into a proper loss and test it; if you want to drop it, leave `w_delta_a = 0`.

---

## 4. Cross-cutting concerns

### 4.1 Unused parameters

* `actor_critic_DKwmp.__init__` accepts `prop_dim=69`, `num_history=5`, `use_observation_function=True`, `w_delta_s=0.0`, `w_delta_a=0.0`, but only `dk_latent_dim` is propagated to `DeepKoopman`. `w_delta_s`, `w_delta_a` and `use_observation_function` are stored but never used. The author's own TODO says "上层传参 num_history, prop_dim" — they never finished plumbing these through.
* The hard-coded `num_actions=29` appears in `state_estimator_DK.encoder.__init__` and `decoder.__init__` — not 12, not configurable. Any non-G1 use (e.g. swapping to a 23-DoF humanoid) would silently break.

### 4.2 Inconsistency between `g1_config.py` and `g1_amp_config.py`

WMP-g1 ships **two** config files for G1:

* `g1_config.py` (no AMP), with `num_dofs=29`, `prop_dim=98`, uses `datasets/g1/walk.csv` directly.
* `g1_amp_config.py` (with AMP), with `num_dofs=29`, `prop_dim=69+29=98`, uses `glob('datasets/g1/*')`.

The two configs disagree on **which env class** is used (`g1_envs.base.legged_robot_g1.LeggedRobotG1` vs `g1.base.legged_robot_g1.LeggedRobotG1`). The repo also has `g1_envs/` AND `g1/` directories with duplicated base classes. This is **not** a clean fork; it is mid-refactor code.

### 4.3 World-model dimensions

README 7.17 says "wm的prop修改成12维". The actual `dreamer/networks.py` is unchanged from upstream — there is no 12-dim prop branch. The README is again out of sync with the code.

### 4.4 `g1_29dof_zy.urdf` does not exist publicly

The config hard-codes `file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_zy.urdf'`. The Unitree G1 official package only ships `g1_29dof.urdf`, `g1_29dof_rev_1_0.urdf`, `g1_29dof_lock_waist.urdf`, etc. The `zy` variant is not documented and not in the upstream Unitree repo. The author likely renamed or hand-edited a URDF and did not commit it. **cleanWMPg1 falls back to `g1_29dof.urdf`** and verifies the joint order matches.

---

## 5. Summary table

| Module | Belongs to WMP paper? | Implementation correctness | CleanWMPg1 carries over? |
|---|---|---|---|
| DeepKoopman | ❌ (author invention) | Buggy (D1, D2, D5, D8); partially scaffolded (see Y issues) | **No** |
| EMLP sym loss | ❌ (idea from prior work, not WMP) | Buggy (S4 wrong action_mean); missing dep (S1); wrong joint group (S3) | **No** |
| `DK_ymloss` | ❌ (claimed in README) | **Partially scaffolded** (`w_delta_a` action residual wired but `w_delta_s` is dead code; default `w_delta_a = 0` means it is OFF) | **No** |
| env/use_amp flag | n/a (engineering) | OK | **Yes** (cleanWMPg1 reproduces) |
| URDF/config | n/a (engineering) | OK with caveat (URDF name mismatch) | **Yes** (with `g1_29dof.urdf` fallback) |
| G1 AMP loader | n/a (engineering) | OK with caveats (uses vendored `pose3d` etc.) | **Yes** (cleanWMPg1 reimplements without vendored deps) |

---

## 6. If you want to evaluate the DK / sym modules honestly

1. Run a single-seed training of WMP-g1's `g1_DK` (the strongest variant) and a single-seed training of cleanWMPg1's `g1_amp`. Compare:
   - Time-to-threshold tracking reward
   - Final reward
   - Wall-clock per iteration (DK adds encoder+propagate+loss overhead — expect ~30% slower)
2. Ablation: comment out `sym_loss` only, then `DK_total_loss` only, see what each contributes.
3. Verify that the sym_loss computation in `amp_ppo_sym.py` matches the formula in any related paper (the candidate is the EMLP paper, Finzi et al. 2021).

Don't trust the README claims about "ymloss" until you grep for it and confirm.