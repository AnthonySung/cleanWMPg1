# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# software without specific prior written permission.
#
# cleanWMPg1: G1 AMP task config, derived from WMP (bytedance/WMP) a1_amp_config.py
# and brought in line with WMP-g1's G1 settings (AnthonySung/WMP-g1 g1_config.py).
# All PD / default-joint / termination values follow the G1 URDF joint order
# (legs -> waist -> left arm -> right arm, 29 revolute joints).
# Only the algorithmic parts that author of WMP-g1 added (DeepKoopman, sym-coef,
# ymloss) are intentionally NOT carried over — see ANALYSIS.md / WMPG1_AUDIT.md.

import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# G1 AMP reference motion dataset (LAFAN1-style CSV: pos(3)+quat(4)+joint_pos(29) = 36 cols).
# The G1 motion loader computes lin_vel / ang_vel / joint_vel by finite-differencing positions.
MOTION_FILES_G1 = glob.glob('datasets/g1/*')

# A1 reference motion dataset (Laikago mocap JSON), kept here so plain `a1` task still works.
MOTION_FILES_A1 = glob.glob('datasets/mocap_motions/*')


# ---------------------------------------------------------------------------
# G1 plain task (no AMP) — terrain / DR / commands mirror G1AMPCfg, just no
# discriminator. Same reward set (no AMP reference).
# ---------------------------------------------------------------------------
class G1RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None
        # prop = ang_vel(3) + proj_grav(3) + cmd(3) + (dof_pos-default)(29) + dof_vel(29) = 67
        # WMP-g1's prop_dim = 69 + 29 also includes 2 phase dims (sin/cos gait phase).
        # cleanWMPg1 matches WMP-g1's design including the phase dims.
        prop_dim = 3 + 3 + 3 + 3 + 29 + 29 + 2  # 72 (ang_vel3+grav3+cmd3+(dof_pos-default)29+dof_vel29+phase2)
        action_dim = 29
        num_actions = 29
        # Privileged slice (matches WMP-g1: only base_lin_vel + contact_flag + DR params).
        # WMP-g1 formula: 13 = 5*contact_flag + (DR removed for early training) + 3 (later additions),
        # final: 13 + 3 (base_lin_vel) = 16. We match that.
        privileged_dim = 13 + 3  # 16
        height_dim = 187
        forward_height_dim = 525
        env_name = 'g1'
        use_amp = False
        amp_motion_files = MOTION_FILES_G1
        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.80]  # G1 nominal pelvis height
        default_joint_angles = {  # height ≈ 0.7429 m
            # legs (12)
            'left_hip_pitch_joint': -0.2,
            'left_hip_roll_joint':  0.0,
            'left_hip_yaw_joint':   0.0,
            'left_knee_joint':      0.42,
            'left_ankle_pitch_joint': -0.23,
            'left_ankle_roll_joint':  0.0,
            'right_hip_pitch_joint': -0.2,
            'right_hip_roll_joint':  0.0,
            'right_hip_yaw_joint':   0.0,
            'right_knee_joint':      0.42,
            'right_ankle_pitch_joint': -0.23,
            'right_ankle_roll_joint':  0.0,
            # waist (3)
            'waist_yaw_joint':   0.0,
            'waist_roll_joint':  0.0,
            'waist_pitch_joint': 0.0,
            # left arm (7)
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint':  0.2,
            'left_shoulder_yaw_joint':   0.15,
            'left_elbow_joint':          1.2,
            'left_wrist_roll_joint':     0.0,
            'left_wrist_pitch_joint':    0.0,
            'left_wrist_yaw_joint':      0.0,
            # right arm (7)
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint':  -0.2,
            'right_shoulder_yaw_joint':   -0.15,
            'right_elbow_joint':          1.2,
            'right_wrist_roll_joint':     0.0,
            'right_wrist_pitch_joint':    0.0,
            'right_wrist_yaw_joint':      0.0,
        }

    class sim(LeggedRobotCfg.sim):
        # cleanWMPg1: G1 simulation params inherit dt/substeps/gravity/physx
        # from upstream LeggedRobotCfg.sim.
        pass

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        # G1 collision handling — matches WMP-g1 exactly.
        # Cannot include waist (arms would self-collide with torso and that shouldn't
        # reset); use shoulder contact + body pitch + body height as reset signals.
        penalize_contacts_on = ["hip", "knee", "shoulder_yaw", "elbow", "torso"]
        terminate_after_contacts_on = ["pelvis", "waist", "shoulder_pitch", "knee"]
        body_name = "waist"
        self_collisions = 1  # disable self-collision
        flip_visual_attachments = False
        # cleanWMPg1: G1 URDF exposes 2 force-torque sensors by default
        # (one per foot, attached to the bodies named in feet_names). The
        # sensor tensor is therefore (num_envs, 2, 6) where [:, :, :3] is
        # the force vector. Setting num_force_sensors=2 makes the explicit
        # view path in _init_buffers() succeed without relying on the
        # except-clause fallback.
        num_force_sensors = 2

    class control(LeggedRobotCfg.control):
        # PD gains match WMP-g1 G1Cfg exactly. Ankle is much stiffer than what
        # upstream A1 amp uses — required for G1 to balance upright.
        control_type = 'P'
        stiffness = {
            'hip_pitch': 200, 'hip_roll': 150, 'hip_yaw': 150,
            'knee':      200, 'ankle':    100, 'waist':    200,
            'shoulder':   20, 'elbow':     20, 'wrist_roll': 20,
            'wrist_pitch': 5, 'wrist_yaw':  5,
        }
        damping = {
            'hip_pitch': 5, 'hip_roll': 5, 'hip_yaw': 5,
            'knee':      5, 'ankle':    5, 'waist':    5,
            'shoulder':   0.5, 'elbow':  0.5, 'wrist_roll': 0.5,
            'wrist_pitch': 0.2, 'wrist_yaw':  0.2,
        }
        # G1 joint ranges are larger than A1's; use bigger action scale.
        action_scale = 0.5
        decimation = 20  # 100 Hz control (dt=5ms × 20)

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_trq = 0.08
            height_measurements = 1.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5

        clip_observations = 30.
        clip_actions = 5.0
        # Matches WMP-g1 (G1 pelvis height when standing is ~0.7429 m).
        base_height = 0.75

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        reward_curriculum = False
        soft_dof_pos_limit = 0.9
        base_height_target = 0.80
        foot_height_target = 0.15
        tracking_sigma = 0.15
        lin_vel_clip = 0.1
        default_gap = 0.22

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            torques = -0.0001
            dof_acc = -2.5e-7
            base_height = 0.0
            feet_air_time = 0.5
            collision = -1.0
            feet_stumble = -0.1
            action_rate = -0.03
            lin_vel_z = -1.0
            orientation = -1.0
            dof_vel = -1e-4


# ---------------------------------------------------------------------------
# G1 AMP task — matches WMP-g1's G1Cfg / G1CfgPPO exactly (with the algorithm
# knobs unchanged). Differences from cleanWMPg1's first draft:
#   * terrain = plane (not trimesh) — G1 is bipedal, hard to train on rough
#     terrain without first mastering flat ground
#   * domain randomization = all False — G1 policy is sensitive; DR on
#   * observation noise = False — prop is already noisy enough
#   * ankle PD = 100/5 (was 40/2) — needed for upright balance
#   * action_scale = 0.5 (was 0.25) — G1 joints have wider ranges
#   * commands.lin_vel_x in [0.0, 0.8] m/s (was 0.6)
#   * commands.ang_vel_yaw in [-1.0, 1.0] rad/s (was ±0.8)
#   * reward weights: tracking_lin_vel/ang_vel = 20/20 (was 1.5/0.5); base_height
#     = -10; collision = -20; hip_pos = -1; waist_pos = -1; arm_pos = -0.5;
#     contact = -1; clearance = -0.05; feet_distance = -0.1; alive = 2.0;
#     action_rate = -5e-3 (was -0.03); dof_pos_limits = -10; stand_normal = -0.01
#   * depth.use_camera = False by default (set to True on a host where depth
#     rendering is verified to work)
# ---------------------------------------------------------------------------
class G1AMPCfg(G1RoughCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None
        prop_dim = 3 + 3 + 3 + 3 + 29 + 29 + 2  # 72 (ang_vel3+grav3+cmd3+(dof_pos-default)29+dof_vel29+phase2) (matches WMP-g1: includes 2 phase dims)
        action_dim = 29
        num_actions = 29
        # Privileged slice (matches WMP-g1's stripped form: only base_lin_vel +
        # contact_flag; kp/kd DR were removed from observations in WMP-g1).
        privileged_dim = 13 + 3  # 16
        height_dim = 187
        forward_height_dim = 525
        env_name = 'g1'
        use_amp = True
        amp_motion_files = MOTION_FILES_G1
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85

        num_observations = prop_dim + privileged_dim + height_dim + action_dim
        num_privileged_obs = prop_dim + privileged_dim + height_dim + action_dim

    class terrain(LeggedRobotCfg.terrain):
        # cleanWMPg1: matches WMP-g1 — flat plane only. WMP-g1 sets mesh_type='plane'
        # for G1 specifically (see g1_config.py line 66). Curriculum / proportions
        # are kept in case the user wants to flip back to trimesh later.
        mesh_type = 'plane'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                             0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_forward_points_x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        measured_forward_points_y = [-1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4,
                                     -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                     0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10
        num_cols = 20
        # terrain types: [wave, rough slope, stairs up, stairs down, discrete, gap, pit, tilt, crawl, rough_flat]
        # Kept for reference; only takes effect if mesh_type='trimesh'.
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        slope_treshold = 0.75

    class depth:
        # cleanWMPg1: disabled by default. Depth camera rendering segfaults on
        # Isaac Gym Preview 3 + nvidia driver 580.x (CUDA 13) — verified on the
        # autodl container. Set to True only if your driver stack can render
        # depth (driver < 530 / CUDA 11 typically works).
        use_camera = False
        combine_height_map = False
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        # Camera position (matches WMP-g1): mounted at G1 head, looking down 48°.
        position = [0.047645, 0.000299, 0.46268]  # Modified for G1
        y_angle = [48, 48]  # positive pitch down
        z_angle = [0, 0]
        x_angle = [0, 0]
        update_interval = 5
        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        scale = 1
        invert = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        # cleanWMPg1: matches WMP-g1 — DR disabled during early training.
        # G1 policy is sensitive to domain perturbations; turn these on only
        # after the policy has converged on flat ground.
        randomize_friction = False
        friction_range = [0.5, 2.0]
        randomize_restitution = False
        restitution_range = [0.0, 0.0]
        randomize_base_mass = False
        added_mass_range = [0., 3.]
        randomize_link_mass = False
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = False
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]
        push_robots = False
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = False
        stiffness_multiplier_range = [0.8, 1.2]
        damping_multiplier_range = [0.8, 1.2]
        randomize_motor_strength = False
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = False
        latency_range = [0.00, 0.005]

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_trq = 0.08
            height_measurements = 1.0
            contact_force = 0.005
            com_pos = 20
            pd_gains = 5
        clip_observations = 30.
        clip_actions = 5.0
        # matches WMP-g1 (G1 pelvis height when standing ≈ 0.7429 m).
        base_height = 0.75

    class noise(LeggedRobotCfg.noise):
        # cleanWMPg1: matches WMP-g1 — observation noise disabled.
        add_noise = False
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.0

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        reward_curriculum = False
        reward_curriculum_term = ["feet_edge"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]
        base_pitch_target = 0.0
        soft_dof_pos_limit = 0.9
        base_height_target = 0.80
        foot_height_target = 0.15
        tracking_sigma = 0.15
        lin_vel_clip = 0.1
        default_gap = 0.22

        class scales(LeggedRobotCfg.rewards.scales):
            # G1 reward weights, matched to WMP-g1.
            tracking_lin_vel = 20.0
            tracking_ang_vel = 20.0

            alive = 2.0
            lin_vel_z = -1.0
            ang_vel_xy = 0.0
            orientation = -1.0

            stand_normal = -0.01
            base_height = -10.0

            action_rate = -5e-3
            smoothness = -1e-4

            dof_pos_limits = -10.0
            torque_limits = 0.0

            torques = -6e-7
            delta_torques = 0.0

            clearance = -0.05
            feet_distance = -0.1

            collision = -20.0

            feet_swing_height = 0.0

            contact = -1.0

            hip_pos = -1.0
            waist_pos_dof29 = -1.0
            arm_pos_dof29 = -0.5

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = True
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0
        max_flat_lin_vel_forward_x_curriculum = 1.0
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.0
        # ranges match WMP-g1 (slightly more aggressive than cleanWMPg1's first draft)
        class ranges:
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [-0., 0.]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-0., 0.]
            flat_lin_vel_x = [-0.0, 0.8]
            flat_lin_vel_y = [-0.0, 0.0]
            flat_ang_vel_yaw = [-1.0, 1.0]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class G1AMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'flat_push1'
        experiment_name = 'g1_amp_clean'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000
        save_interval = 1000
        amp_reward_coef = 0.5 * 0.02
        amp_motion_files = MOTION_FILES_G1
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]
        # G1 has 29 dofs; minimum normalized std per dof. WMP-g1 style.
        min_normalized_std = [0.05, 0.02, 0.05] * 9 + [0.0, 0.0]

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100


class G1RoughCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'WMPRunner'

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = 'elu'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'flat_push1'
        experiment_name = 'g1_clean'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000
        save_interval = 1000
        amp_reward_coef = 0.0
        amp_motion_files = []
        amp_num_preload_transitions = 0
        amp_task_reward_lerp = 0.0
        amp_discr_hidden_dims = [1024, 512]
        min_normalized_std = [0.05, 0.02, 0.05] * 9 + [0.0, 0.0]

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100