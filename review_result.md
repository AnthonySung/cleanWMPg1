Code audit complete. The review is saved to `review_result.md`. Here's the high-level summary:

## Ratings

| File | Rating | Key Findings |
|------|--------|--------------|
| `g1_amp_config.py` | 3/5 | **P0**: `privileged_dim=16` likely doesn't match actual layout |
| `g1_motion_loader.py` | 3/5 | **P0**: `sample_motion_for_init` shape mismatch (N,36) vs 71-dim frames |
| `amp_ppo.py` | 5/5 | `use_amp` guard verified clean at all 5 sites |
| `wmp_runner.py` | 4/5 | `env.reset()` move and adaptive history dim correct; slicing fragility from Bug 1.1 |
| `legged_robot.py` | 3/5 | **4 P1 bugs** in G1-specific rewards (arm/waist/contact indexing, dead ankle_pos) |
| `legged_robot_config.py` | 4/5 | `num_force_sensors=4` fallback works; should be 0 for G1 |
| `actor_critic_wmp.py` | 5/5 | PT 1.10 compat clean (uses `.mean` and `.stddev`, not `.mode`/`.std`) |

## Top-3 Risks

1. **`privileged_dim=16` mismatch** (g1_amp_config.py:50). Actual privileged prefix = `contact_flag(N) + contact_force(12)`. With `penalize_contacts_on` substring-matching G1 body names, N≈13 → actual prefix ≈25 dims. This breaks every offset-based slice in `wmp_runner.py:117,232,265` and `actor_critic_wmp.py:205,228`. The auto-fix in `compute_observations` updates `num_obs` but **not** `privileged_dim`.

2. **`sample_motion_for_init` shape bug** (g1_motion_loader.py:218). Allocates `(N, 36)` but assigns 71-dim tensors → RuntimeError at training start. `G1AMPCfg` sets `reference_state_initialization=True` with prob 0.85, so this WILL fire. Additionally, `legged_robot.py` still imports the A1 `AMPLoader` (not `AMPLoaderG1`), so even with the shape fixed the slicing in `AMPLoader.get_joint_pose_batch` etc. would still mismatch.

3. **G1 reward indexing drift** (legged_robot.py:1395-1438).
   - `_reward_arm_pos_dof29` `[13:27]` is asymmetric (covers 7 left arm + 5 right arm + 2 waist, missing `right_wrist_pitch/yaw`)
   - `_reward_waist_pos_dof29` `[12:13]` covers only `waist_yaw` (1 of 3 waist DoFs)
   - `_reward_not_fly`/`_reward_contact` index `contact_flag[:, 0:2]` which are hip/knee contacts, not foot contacts
   - `_reward_stand_normal` is byte-for-byte identical to `_reward_orientation`

No project files were modified.
d the actor's `obs[:, privileged_dim+6:privileged_dim+9]` command slicing (actor_critic_wmp.py ~205). |
| 1.2 | **P1** | line ~52 (`# WMP-g1 formula: 13 = 5*contact_flag + ...`) | The comment is mathematically inconsistent — `5*contact_flag` would be 5, not 13. The justification for the magic number 13 is undocumented. |
| 1.3 | **P1** | line ~50–51 (`prop_dim = 3+3+3+3+29+29+2 # 72`) | The comment enumerates 6 components but lists only 5 in the `# ang_vel3+grav3+cmd3+(dof_pos-default)29+dof_vel29+phase2` annotation (sums to 69, not 72). The leading `3` (base_lin_vel) is missing from the annotation. Code is correct; comment is incomplete. |
| 1.4 | **P2** | line ~159 (`reward_curriculum_term = ["feet_edge"]`) | `feet_edge` is in the curriculum term list, but `feet_edge` is **not** in `rewards.scales`. Curriculum logic in `compute_reward` (`legged_robot.py`) does `if(name == term[j])`, so this curriculum is silently dead. |
| 1.5 | **P2** | line ~96 (`body_name = "waist"`) | Substring match — for G1 URDF with 3 waist links (`waist_yaw_link`, `waist_roll_link`, `waist_pitch_link`) this may or may not work depending on downstream consumer. Verify against `resources/robots/g1_description/g1_29dof.urdf`. |

### Verification of dimension arithmetic

- `prop_dim = 3+3+3+3+29+29+2 = 72` ✓ (base_lin_vel + base_ang_vel + proj_gravity + commands + dof_pos + dof_vel + phase)
- `action_dim = 29` ✓
- `num_actions = 29` ✓
- `privileged_dim = 16` ⚠ (only matches layout if `contact_flag` has exactly 4 dims — see 1.1)
- `height_dim = 187` (17 × 11) ✓
- `forward_height_dim = 525` (21 × 25) ✓
- `num_observations = 72 + 16 + 187 + 29 = 304` (will be auto-fixed to ~311 at runtime — see Bug 1.1)
- PD gains: ankle stiffness/damping 100/5 (matches WMP-g1, plausible for G1 balancing) ✓

---

## File 2 — `rsl_rl/datasets/g1_motion_loader.py`

**Rating: 3/5**

### Bugs

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| 2.1 | **P0** | line ~218 (`sample_motion_for_init`) | Allocates `out = torch.zeros(num_samples, self.TOTAL_COLS=36, device=...)` but assigns `out[i] = self._get_full_frame_at_time(int(ti), t)` which returns a **71-dim** vector (pos+rot+joint_pos+lin_vel+ang_vel+joint_vel). Shape mismatch → `RuntimeError` on assignment when `reference_state_initialization=True`. `G1AMPCfg` sets `reference_state_initialization = True` with prob 0.85, so this WILL trigger at training start. |
| 2.2 | **P2** | line ~245 (`_infer_dt`) | Opens the file twice — once to read the head line, then again to `json.loads(open(path).read())`. Should reuse the file handle. Also: when the header IS valid JSON, the entire CSV body is silently discarded (`obj = json.loads(open(path).read())` would fail on CSV). The `try/except` masks the failure as "default dt", but a CSV file that accidentally starts with `{` will be misread. |
| 2.3 | **P2** | line ~169 (`_get_full_frame_at_time`) | Uses linear interpolation between adjacent frames for **all** 71 dims, including the quaternion (rot) at indices 3:7. Should use `slerp` for quaternion; linear lerp on a unit quaternion only produces a unit quaternion at the endpoints. At 30 Hz the error is small but accumulates across long sequences. |
| 2.4 | **P2** | line ~158 (`trajectories_full.pop()` after append) | Trajectory is fully constructed (with velocities from finite-difference) and then popped if too short. This wastes compute and the `trajectory_frame_durations` is left with a stale entry for that path (though `trajectories_full` is correctly emptied — `trajectory_frame_durations` is keyed by file order and would mismatch later indices if any file were skipped, but `_get_full_frame_at_time` uses `trajectory_frame_durations[traj_idx]` which now points at the wrong file). Actually verified: since we pop the trajectory but keep `trajectory_frame_durations` in the same loop, downstream `_get_full_frame_at_time` indexing is broken if any short file appears between two valid files. |

### Verification

- `AMP observation_dim = 29 + 3 + 3 + 29 = 64` ✓ (joint_pos + lin_vel + ang_vel + joint_vel)
- Quaternion math (`_quat_mul`, `_quat_conj`, `_rotate_vec_by_quat`) ✓ correct Hamilton product in xyzw convention
- Finite-difference for `lin_vel`/`joint_vel` ✓ (forward Euler, missing terminal frame zero-filled — minor)
- Angular velocity conversion (world → body frame via `q^*`) ✓ correct
- `feed_forward_generator` strips `s[..., 7:]` (pos+rot = 7 dims) → 64 dims ✓

---

## File 3 — `rsl_rl/algorithms/amp_ppo.py`

**Rating: 5/5** — no bugs found.

### `use_amp` guard verification

All five sites that touch AMP-specific state correctly gate on `self.use_amp`:

| Site | Line | Guard |
|------|------|-------|
| `__init__` (discriminator + amp_storage + amp_data + amp_normalizer + amp_transition) | ~76–93 | `if self.use_amp: ... else: discriminator/amp_data/... = None` ✓ |
| `act` (records `amp_transition.observations`) | ~131 | `if self.use_amp:` ✓ |
| `process_env_step` (inserts into `amp_storage`) | ~141 | `if self.use_amp:` ✓ |
| `update` (builds `amp_policy_generator`/`amp_expert_generator`, computes `amp_loss`/`grad_pen_loss`) | ~165–251 | `if self.use_amp: ... else: ... = None` ✓ |
| Loss accumulation (`mean_amp_loss`, `mean_policy_pred`, `mean_expert_pred`) | ~263–267 | `if self.use_amp and isinstance(amp_loss, torch.Tensor) else 0.0` ✓ |

The fallback `amp_loss = 0; grad_pen_loss = 0` (int zero) correctly adds to the loss tensor when AMP is disabled. The discriminator prediction step (called from runner) is gated by `if self.alg.use_amp:` in `wmp_runner.py` line ~263 — verified clean.

---

## File 4 — `rsl_rl/runners/wmp_runner.py`

**Rating: 4/5**

### Bugs

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| 4.1 | **P1** | line ~117 (`self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim-3)`) | Relies on `self.env.privileged_dim == 16` matching the actual obs-buf layout. If `contact_flag` has any dim ≠ 4 (likely 13 for G1), the history slice covers the wrong range. (Downstream consequence of Bug 1.1.) |
| 4.2 | **P1** | line ~232 and ~265 (slicing `obs[:, self.env.privileged_dim:self.env.privileged_dim + 6]` and `obs[:, self.env.privileged_dim + 9:-self.env.height_dim]`) | Same root cause as 4.1: the magic offsets assume the layout. If `privileged_dim` is wrong, `obs_without_command` includes parts of the privileged slice instead of `base_lin_vel/base_ang_vel/dof_pos/dof_vel`. The auto-fix in `compute_observations` updates `num_obs`/`num_privileged_obs` but **does not** update `privileged_dim` or the slicing offsets. |
| 4.3 | **P2** | line ~158 (`_ENV_LOADERS` dispatch) | Dispatch uses `getattr(self.env.cfg.env, 'env_name', 'a1')` with `'a1'` as fallback. If someone runs a config without `env_name` set, the runner silently loads the A1 loader and `dreamer/configs.yaml` for a G1 robot — would fail later but at a confusing location. Recommend explicit error. |
| 4.4 | **P2** | line ~427 (`os.system("cp ./legged_gym/envs/a1/a1_amp_config.py " + self.log_dir + "/")`) | Hardcoded `a1/` path — should be `g1/` for G1 runs. Also: `os.system` with `cp` is not Windows-safe; if the project is ever run on Windows this breaks. |

### Positive findings

- `env.reset()` moved to **before** `history_dim` computation ✓ (line ~98). This is correct: the auto-fix in `compute_observations` only fires after the first `env.step()`, so `num_obs` must be stabilised before the runner computes `history_dim`.
- `trajectory_history` adaptive dim via `obs_without_command.shape[-1]` ✓ (line ~238) — recomputes on every iteration; consistent with the `history_dim` formula in `__init__` when layout is correct.
- `_ENV_LOADERS` dict cleanly separates `a1`/`g1` loader + dreamer yaml paths ✓.

---

## File 5 — `legged_gym/envs/base/legged_robot.py`

**Rating: 3/5**

### Bugs

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| 5.1 | **P1** | line ~1395 (`_reward_ankle_pos`) | Function defined but **never registered** — `ankle_pos` is missing from `G1AMPCfg.rewards.scales`. `_prepare_reward_function` only iterates over keys in `self.reward_scales`, so this reward is silently dead. |
| 5.2 | **P1** | line ~1407 (`_reward_waist_pos_dof29`) — slice `[12:13]` | Only 1 dim (waist_yaw at index 12). The function name implies all 3 waist DoFs (yaw/roll/pitch at 12/13/14). Either rename to `waist_yaw_pos` or expand slice to `[12:15]`. The comment claims this matches WMP-g1; if so, WMP-g1 itself has the same mismatch. |
| 5.3 | **P1** | line ~1413 (`_reward_arm_pos_dof29`) — slice `list(range(13, 27))` | With G1 URDF order (legs 0-11, waist 12-14, left arm 15-21, right arm 22-28), `[13:27]` includes `waist_roll` (13), `waist_pitch` (14), 7 left arm joints (15-21), and only 5 right arm joints (22-26, missing `right_wrist_pitch` 27 and `right_wrist_yaw` 28). Asymmetric — penalizes left arm more than right. Should be `list(range(15, 29))` to cover all 14 arm joints, or `list(range(13, 29))` if you want waist+arms (matches `upper_action_rate` slice). |
| 5.4 | **P1** | line ~1410 (`_reward_not_fly`) and ~1417 (`_reward_contact`) | Both gate on `self.contact_flag` having ≥ 2 dims, but `contact_flag` is built from `penalised_contact_indices`, which for G1 has ~13 entries (not 2). `contact_flag[:, 0]` and `contact_flag[:, 1]` correspond to hips (hip_pitch-ish substring matches), **not** feet. These rewards fire on hip contacts, not foot contacts — produces the wrong imitation signal. |
| 5.5 | **P2** | line ~1416 (`_reward_stand_normal`) vs line ~1178 (`_reward_orientation`) | Both compute `torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)` — byte-for-byte identical. With `orientation=-1.0` AND `stand_normal=-0.01` enabled, the policy is effectively penalized `-1.01` for body tilt. Either drop one or differentiate (e.g., `stand_normal` only applies when commands ≈ 0). |
| 5.6 | **P2** | line ~1438 (`_reward_upper_action_rate`) — slice `[13:]` | Covers 16 dims (waist_roll + waist_pitch + 14 arm). Inconsistent with `_reward_arm_pos_dof29` (which covers `[13:27]` = 14 dims). The action-rate penalty and the position penalty therefore don't align. |
| 5.7 | **P2** | line ~1460 (and again at the very end of the file, ~line ~1670) | `_reward_delta_torques` is **defined twice** with identical bodies. Python silently keeps the last definition; no functional impact but indicates a sloppy merge. |

### Positive findings

- `num_force_sensors` fallback (line ~548) correctly handles the G1 case: `if force_sensor_readings is None or ... numel() == 0: self.sensor_forces = torch.zeros(num_envs, num_sensors, 3, ...)` then a `try/except` on `.view(num_envs, num_sensors, 6)` for the A1 case. ✓
- Auto-fix in `compute_observations` (line ~466) correctly catches the `contact_flag` cardinality mismatch and updates `num_obs`/`num_privileged_obs`. ✓ (But does not fix `privileged_dim` — see 4.1, 4.2.)
- `reference_state_initialization` path is broken (see 2.1) but the wrapper logic in `reset_idx` and `_reset_dofs_amp` itself is fine.
- Action latency logic (line ~159) cleanly integrates with `random.randint(rng[0], rng[1])` per-step. ✓

---

## File 6 — `legged_gym/envs/base/legged_robot_config.py`

**Rating: 4/5**

### Bugs

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| 6.1 | **P2** | line ~175 (`num_force_sensors = 4`) | Hardcoded 4, but G1 URDF has 0 force sensors (and A1 has 4). The fallback path in `legged_robot.py` (`hasattr(self.cfg.asset, 'num_force_sensors') else 4`) means this value is used even for G1 — `contact_force` in `compute_observations` is a `(num_envs, 12)` zero tensor for G1, not a 12-dim meaningful signal. Recommend `num_force_sensors = 0` for G1 or remove the field and detect from URDF. |
| 6.2 | **P2** | line ~157 (`reward_curriculum_schedule = [[0, 1000, 1.0, 0.0]]`) | List of one 4-tuple. Code in `legged_robot.__init__` and `update_reward_curriculum` accesses `schedule[2]` and `schedule[i][0..3]`. Single-element list is correct shape. The comment in the file says it was changed from "single 4-tuple" → "list-of-schedules", which is the right move. ✓ |

### Positive findings

- `num_force_sensors` fallback in `_init_buffers` is correct ✓.
- Added reward-related defaults (`lin_vel_clip`, `foot_height_target`, `base_height_target`, `default_gap`) match the WMP upstream reads; missing-field errors prevented. ✓
- Per-axis `com_*_pos_range` correctly added (WMP `legged_robot.py` reads `com_x_pos_range` etc., upstream had only `com_pos_range`). ✓

---

## File 7 — `rsl_rl/modules/actor_critic_wmp.py`

**Rating: 5/5**

### Bugs

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| 7.1 | **P2** | line ~197 (`return self.distribution.mean`) | Property correctly exposes `mean` (not `mode`) — PT 1.10 compatible. ✓ No bug. Note: `Normal.mode()` is also `mean`, so this is robust either way, but `mean` is the convention. |

### Positive findings

- `action_mean` → `self.distribution.mean` ✓ (PT 1.10 compat)
- `action_std` → `self.distribution.stddev` ✓ (PT 1.10 compat — `Normal.std` is a method in some versions)
- `Normal.set_default_validate_args = False` ✓ (perf optimization)
- `update_distribution` correctly broadcasts `std` to all action dims via `mean * 0. + std` ✓
- History encoder, wm encoder, actor, critic are all clean MLPs ✓
- `get_linear_vel` correctly slices `latent_vector[:, -3:]` (last 3 dims are the vel prediction head) ✓

---

## Top-3 Risks (prioritised)

1. **`privileged_dim=16` mismatch (Bug 1.1)** — affects **every** downstream consumer that slices `obs[:, privileged_dim:...]`:
   - `wmp_runner.py:117` — `history_dim` formula
   - `wmp_runner.py:232, ~265` — `obs_without_command` slicing
   - `actor_critic_wmp.py:205, ~228` — `command` slicing in `act`/`act_inference`
   
   The auto-fix in `legged_robot.py:466` updates `num_obs`/`num_privileged_obs` but **not** `privileged_dim`, so the offsets stay wrong. The most likely actual value for G1 is **25** (12 contact_force + 13 contact_flag from 5×penalize_contacts_on entries). Fix: either reduce `penalize_contacts_on` to 4 entries so `N=4` matches the configured 16 (= 12 + 4), or change `privileged_dim=25` and the slicing offsets to derive from the actual `contact_flag` cardinality.

2. **`sample_motion_for_init` shape bug (Bug 2.1)** — allocates `(N, 36)` but assigns 71-dim frames; will crash at training start because `G1AMPCfg` enables `reference_state_initialization=True` with `prob=0.85`. Fix: change `out` to `(num_samples, 71)` to match `_get_full_frame_at_time` output, **or** also update `legged_robot._reset_dofs_amp` and `_reset_root_states_amp` to use the new G1 layout (the env currently still imports the **A1** `AMPLoader`, not `AMPLoaderG1`, so even if the shape is fixed the slicing in `AMPLoader.get_joint_pose_batch` etc. would still be wrong).

3. **G1 reward indexing drift (Bugs 5.2, 5.3, 5.4, 5.5)** — three of the ported rewards target the wrong joints:
   - `_reward_arm_pos_dof29` asymmetric: penalizes left arm fully (7 joints) but only 5 of 7 right arm joints, plus 2 waist joints.
   - `_reward_waist_pos_dof29` covers only `waist_yaw` (1 of 3 waist DoFs).
   - `_reward_not_fly` / `_reward_contact` index `contact_flag[:, 0:2]` which are hip/knee contacts, not foot contacts — produces a wrong imitation signal.
   - `_reward_stand_normal` duplicates `_reward_orientation` exactly.
   
   These will train the policy on a corrupted reward signal; AMP imitation may still work but the policy won't track the intended joint poses.

---

## Other actionable issues

- **Dead reward functions** (Bug 5.1): `_reward_ankle_pos`, `_reward_not_fly`, `_reward_upper_action_rate`, `_reward_upper_action_smoothness` are defined but not registered in `G1AMPCfg.rewards.scales`. If you intended them to be active, add the corresponding scale entries.
- **Hardcoded `a1/` path** (Bug 4.4): `os.system("cp ./legged_gym/envs/a1/a1_amp_config.py ...")` in `wmp_runner.py:427`. Should be `g1/g1_amp_config.py` for G1 runs, and `shutil.copy` would be portable.
- **`_infer_dt` opens file twice** (Bug 2.2): minor I/O efficiency + edge case where a CSV file whose first line starts with `{` (rare) gets misread.
- **Quaternion lerp instead of slerp** (Bug 2.3): small precision issue, not a blocker.

---

## Files NOT modified

No project files were changed during this audit. This review is the only artifact written (`review_result.md`).