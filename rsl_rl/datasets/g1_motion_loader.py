# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# cleanWMPg1: AMP motion loader for G1 humanoid reference motions.
#
# Format (per row of the CSV, 36 columns):
#   [0:3]   root_pos  (x, y, z) in world frame, metres
#   [3:7]   root_rot  (qx, qy, qz, qw) quaternion in world frame
#   [7:36]  joint_pos (29) joint angles, radians, in URDF order
#
# Lin_vel / ang_vel / joint_vel are NOT provided directly; we finite-difference
# successive frames. Frame rate is read from the optional first-line JSON header
# `"FrameDuration"` if present, otherwise assumed to be 30 Hz (matching the
# LAFAN1-style G1 dataset shipped in cleanWMPg1).
#
# Quaternion math is implemented with torch primitives (no third-party
# transforms library) so this loader is self-contained.

import glob
import json
import numpy as np
import torch

AMP_TIME_BETWEEN_FRAMES_DEFAULT = 1.0 / 30.0


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product (xyzw convention)."""
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _rotate_vec_by_quat(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """v_world = q * (0, v) * q^*.  q is xyzw."""
    qv = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., :3]


class AMPLoaderG1:
    """AMP motion dataset loader for G1.

    Compatible shape with the upstream `AMPLoader` so that AMPDiscriminator and
    AMPPPO can consume the resulting trajectories without changes.

    After loading, the relevant state for AMP-style adversarial imitation is:
        amp_obs[t]   = [joint_pos(t),  base_lin_vel(t),  base_ang_vel(t),  joint_vel(t)]
        amp_obs[t+1] = same with (t+1)
    """

    # Column layout for the 36-column G1 CSV
    POS_SIZE = 3
    ROT_SIZE = 4
    JOINT_POS_SIZE = 29   # G1 with 29 revolute DoF
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 29
    TOTAL_COLS = POS_SIZE + ROT_SIZE + JOINT_POS_SIZE  # 36 (CSV format, used for I/O)
    # Full state dim stored in trajectories_full (pos + rot + joint_pos + lin_vel + ang_vel + joint_vel):
    FULL_STATE_DIM = POS_SIZE + ROT_SIZE + JOINT_POS_SIZE + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + JOINT_VEL_SIZE  # 71

    def __init__(self,
                 device,
                 time_between_frames,
                 motion_files,
                 data_dir='',
                 preload_transitions=True,
                 num_preload_transitions=2_000_000,
                 frame_duration_default=AMP_TIME_BETWEEN_FRAMES_DEFAULT,
                 min_traj_length_s=0.5):
        self.device = device
        self.time_between_frames = time_between_frames
        self.motion_files = list(motion_files)
        self.frame_duration_default = frame_duration_default

        self.trajectories_full = []     # full state per frame: pos+rot+joint_pos+lin_vel+ang_vel+joint_vel
        self.trajectory_lens = []       # seconds
        self.trajectory_weights = []    # sampling weights
        self.trajectory_frame_durations = []

        if not self.motion_files:
            print("[AMPLoaderG1] WARNING: no motion files provided.")
            return

        for path in self.motion_files:
            try:
                raw = np.loadtxt(path, delimiter=',')
            except Exception as e:
                print(f"[AMPLoaderG1] failed to load {path}: {e}")
                continue
            if raw.ndim == 1:
                raw = raw[None, :]
            if raw.shape[1] != self.TOTAL_COLS:
                print(f"[AMPLoaderG1] skip {path}: expected {self.TOTAL_COLS} cols, got {raw.shape[1]}")
                continue

            pos = torch.as_tensor(raw[:, 0:3], dtype=torch.float32, device=device)
            rot = _quat_normalize(torch.as_tensor(raw[:, 3:7], dtype=torch.float32, device=device))
            joint_pos = torch.as_tensor(raw[:, 7:7 + self.JOINT_POS_SIZE],
                                        dtype=torch.float32, device=device)

            dt = self._infer_dt(path, raw.shape[0])
            self.trajectory_frame_durations.append(dt)

            # Compute velocities by finite difference (forward Euler)
            lin_vel = torch.zeros_like(pos)
            ang_vel = torch.zeros_like(pos)
            joint_vel = torch.zeros_like(joint_pos)
            if raw.shape[0] > 1:
                lin_vel[1:] = (pos[1:] - pos[:-1]) / dt
                joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt

                # angular velocity from quaternion difference, expressed in body frame
                rel_rot = _quat_mul(rot[1:], _quat_conj(rot[:-1]))   # q_{t} * q_{t-1}^{-1}
                # angle-axis from relative rotation
                w = rel_rot[..., 3].clamp(-1.0, 1.0)
                angle = 2.0 * torch.atan2(torch.linalg.norm(rel_rot[..., :3], dim=-1), w)
                # small-angle safe normalisation
                sin_half = torch.linalg.norm(rel_rot[..., :3], dim=-1)
                axis = rel_rot[..., :3] / sin_half.clamp(min=1e-8).unsqueeze(-1)
                # Fold >pi rotations into [-pi, pi]
                neg = angle > np.pi
                angle = torch.where(neg, 2 * np.pi - angle, angle)
                axis = torch.where(neg.unsqueeze(-1), -axis, axis)
                world_ang = axis * (angle / dt).unsqueeze(-1)
                # Express in body frame: ang_body = q^* * world_ang * q
                ang_vel[1:] = _rotate_vec_by_quat(world_ang, _quat_conj(rot[1:]))

            traj = torch.cat([pos, rot, joint_pos, lin_vel, ang_vel, joint_vel], dim=-1)
            self.trajectories_full.append(traj)
            traj_len_s = (raw.shape[0] - 1) * dt
            if traj_len_s < min_traj_length_s:
                print(f"[AMPLoaderG1] skip {path}: too short ({traj_len_s:.2f}s < {min_traj_length_s}s)")
                self.trajectories_full.pop()
                continue
            self.trajectory_lens.append(traj_len_s)
            self.trajectory_weights.append(traj_len_s)

        if not self.trajectories_full:
            raise RuntimeError("[AMPLoaderG1] no valid trajectories loaded.")

        self.trajectory_weights = np.asarray(self.trajectory_weights, dtype=np.float64)
        self.trajectory_weights /= self.trajectory_weights.sum()
        self.trajectory_frame_durations = np.asarray(self.trajectory_frame_durations, dtype=np.float64)
        self.trajectory_lens = np.asarray(self.trajectory_lens, dtype=np.float64)
        self.trajectory_num_frames = np.asarray(
            [t.shape[0] for t in self.trajectories_full], dtype=np.float64)
        # cumulative number of frames (used for sampling)
        self._frame_cum = np.cumsum([0] + [t.shape[0] for t in self.trajectories_full])

        # ---- AMP observation dimension ----
        # joint_pos(29) + base_lin_vel(3) + base_ang_vel(3) + joint_vel(29) = 64
        self.observation_dim = (
            self.JOINT_POS_SIZE + self.LINEAR_VEL_SIZE +
            self.ANGULAR_VEL_SIZE + self.JOINT_VEL_SIZE
        )

        # Preload (state, next_state) pairs for PPO updates.
        self.preloaded_s = None
        self.preloaded_s_next = None
        if preload_transitions:
            self.preload(num_preload_transitions)

    # ------------------------------------------------------------------
    def _infer_dt(self, path, num_rows):
        # We try to read the first line as JSON to detect an explicit FrameDuration.
        # If the file is plain CSV, fall back to the default.
        try:
            with open(path, "r") as f:
                head = f.readline()
                if head.lstrip().startswith("{"):
                    obj = json.loads(open(path).read())
                    return float(obj.get("FrameDuration", self.frame_duration_default))
        except Exception:
            pass
        return self.frame_duration_default

    # ------------------------------------------------------------------
    def _get_full_frame_at_time(self, traj_idx, t):
        """Linear-interpolated full state at time t (seconds) within traj_idx."""
        traj = self.trajectories_full[traj_idx]
        dt = self.trajectory_frame_durations[traj_idx]
        f = t / dt
        f0 = int(np.floor(f))
        f1 = min(f0 + 1, traj.shape[0] - 1)
        a = float(f - f0)
        return (1.0 - a) * traj[f0] + a * traj[f1]

    # ------------------------------------------------------------------
    def preload(self, num_transitions):
        print(f"[AMPLoaderG1] preloading {num_transitions} transitions...")
        if not self.trajectories_full:
            return
        traj_idx = np.random.choice(
            len(self.trajectories_full),
            size=num_transitions,
            p=self.trajectory_weights,
        )
        s_list = []
        s_next_list = []
        for i in range(num_transitions):
            ti = traj_idx[i]
            max_t = self.trajectory_lens[ti] - self.time_between_frames
            if max_t <= 0:
                continue
            t = np.random.uniform(0.0, max_t)
            s = self._get_full_frame_at_time(ti, t)
            s_next = self._get_full_frame_at_time(ti, t + self.time_between_frames)
            s_list.append(s)
            s_next_list.append(s_next)
        if not s_list:
            return
        self.preloaded_s = torch.stack(s_list, dim=0)
        self.preloaded_s_next = torch.stack(s_next_list, dim=0)
        print(f"[AMPLoaderG1] preloaded {self.preloaded_s.shape[0]} transitions.")

    def feed_forward_generator(self, num_mini_batches, mini_batch_size):
        """Yields (state, next_state) mini-batches for the discriminator."""
        if self.preloaded_s is None:
            self.preload(num_mini_batches * mini_batch_size)
        N = self.preloaded_s.shape[0]
        for _ in range(num_mini_batches):
            idx = torch.randint(0, N, (mini_batch_size,), device=self.preloaded_s.device)
            s = self.preloaded_s[idx]
            s_next = self.preloaded_s_next[idx]
            # Strip pos(3)+rot(4) -> take joint_pos+lin_vel+ang_vel+joint_vel = 64 dims.
            s_amp = s[..., 7:]
            s_next_amp = s_next[..., 7:]
            yield s_amp, s_next_amp

    # ------------------------------------------------------------------
    def sample_motion_for_init(self, num_samples):
        """Used by reference_state_initialization: sample one frame per env."""
        if not self.trajectories_full:
            return None
        idx = np.random.choice(len(self.trajectories_full),
                               size=num_samples, p=self.trajectory_weights)
        # cleanWMPg1: previously allocated (num_samples, TOTAL_COLS)=36 but
        # _get_full_frame_at_time returns FULL_STATE_DIM=71 entries. Use the
        # correct size here so the assignment doesn't crash.
        out = torch.zeros(num_samples, self.FULL_STATE_DIM, device=self.device)
        for i, ti in enumerate(idx):
            traj = self.trajectories_full[int(ti)]
            t = np.random.uniform(0.0, self.trajectory_lens[int(ti)])
            out[i] = self._get_full_frame_at_time(int(ti), t)
        return out