# WMP-g1 G1 配置 vs cleanWMPg1 G1 配置对比表

> 数据来源:`WMP-g1/legged_gym/envs/g1/g1/g1_config.py`(最新修订版,README 7.17 之后的状态)
>
> 与之对比:`cleanWMPg1/legged_gym/envs/g1/g1/g1_amp_config.py`(纯 G1 适配,无 DK/sym)

## 1. 环境(terrain)

| 项 | WMP-g1 G1Cfg | cleanWMPg1 G1AMPCfg | 评价 |
|---|---|---|---|
| `mesh_type` | **`'plane'`** | `'trimesh'` | **WMP-g1 故意只给 plane**(G1 双足难训,先平地学走路,trimesh 是给 A1 四足用的) |
| `curriculum` | `True` | `True` | 一致(WMP-g1 留 curriculum 是为后面支持地形,但 G1Cfg 用 plane 实际无效) |
| `measure_heights` | `True` | `True` | 一致 |
| `terrain_proportions` | `[0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]` | 同 | plane 模式下不生效 |

**结论**:cleanWMPg1 应该改 `mesh_type='plane'`,把 terrain 调到 G1 友好的初始难度。

## 2. Domain randomization

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `randomize_friction` | **`False`** | `True`(继承父类) | WMP-g1 关掉 DR,因为 G1 双足极敏感,DR 会让早期训练不稳 |
| `randomize_base_mass` | **`False`** | `True` | 同 |
| `randomize_link_mass` | **`False`** | `True` | 同 |
| `randomize_com_pos` | **`False`** | `True` | 同 |
| `randomize_gains` | **`False`** | `True` | 同 |
| `randomize_motor_strength` | **`False`** | `True` | 同 |
| `randomize_action_latency` | **`False`** | `False` | 一致 |
| `push_robots` | **`False`** | `True`(继承) | G1 早期不推 |

**结论**:cleanWMPg1 应该关掉大部分 DR(friction / mass / gains / motor_strength),跟 WMP-g1 一致——除非你专门要测试 DR 在 G1 上的鲁棒性。

## 3. Noise

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `add_noise` | **`False`** | `True`(A1 默认) | WMP-g1 关噪声——G1 关节多,prop 已经够乱,再加观测噪声训练效率低 |

**结论**:cleanWMPg1 应该关掉观测噪声。

## 4. 控制(action_scale / PD)

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `action_scale` | **`0.5`** | `0.25`(继承父类 A1 默认) | G1 关节范围比 A1 大(髋 ±2.18, 膝 -2.66~ -0.34 等),需要更大 scale |
| `decimation` | `20` | `20` ✓ | 一致(100Hz 控制,20×5ms=100ms 决策周期) |
| `stiffness.hip_pitch/roll/yaw` | `200/150/150` | 同 | ✓ |
| `stiffness.knee` | `200` | 同 | ✓ |
| `stiffness.ankle` | **`100`** | `40` | WMP-g1 的 100 比 cleanWMPg1 强 2.5x,适合 G1 站立稳定 |
| `stiffness.waist` | `200` | 同 | ✓ |
| `stiffness.shoulder` | `20` | 同 | ✓ |
| `stiffness.elbow` | `20` | 同 | ✓ |
| `stiffness.wrist_*` | `20/5/5` | 同 | ✓ |
| `damping.*` | `5/5/5/5/5/5/0.5/0.5/0.5/0.2/0.2` | 同 | ✓ |

**结论**:`action_scale=0.5` + `ankle stiffness=100` 是关键差异。cleanWMPg1 应该改。

## 5. Commands(命令空间)

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `num_commands` | `4`(lin_vel_x, lin_vel_y, ang_vel_yaw, heading) | 同 | 一致 |
| `heading_command` | `True` | `True` | 一致 |
| `resampling_time` | `10s` | `10s` | ✓ |
| `lin_vel_x range` | `[0.0, 0.8]` m/s | `[0.0, 0.6]` m/s | WMP-g1 让 G1 跑更快 |
| `lin_vel_y range` | `[-0., 0.]`(不侧移) | 同 | ✓ |
| `ang_vel_yaw range` | `[-1.0, 1.0]` rad/s | `[-0.8, 0.8]` | WMP-g1 更激进 |

**结论**:WMP-g1 给 G1 更大的速度范围。cleanWMPg1 应该匹配(`[0, 0.8]` 和 `[-1.0, 1.0]`)。

## 6. Rewards(关键差异)

| Reward | WMP-g1 权重 | cleanWMPg1(继承 A1 默认) | 评价 |
|---|---|---|---|
| `tracking_lin_vel` | **`20.0`** | `1.5` | G1 必须强力跟速度 |
| `tracking_ang_vel` | **`20.0`** | `0.5` | 同 |
| `alive` | **`2.0`** | **不存在**(我之前删了因为父类没实现) | 必须实现 `_reward_alive` 返回 1.0 |
| `orientation` | `-1.0` | 同 | ✓ |
| `base_height` | **`-10.0`** | `0.0` | G1 必须严格保持高度 0.78m |
| `stand_normal` | `-0.01` | 不存在 | WMP-g1 鼓励站立时身体朝上 |
| `action_rate` | `-5e-3` | `-0.03` | G1 关节多,action_rate 要轻 |
| `dof_pos_limits` | `-10.0` | 不存在 | G1 严防超限 |
| `collision` | **`-20.0`** | `-1.0` | G1 不能撞 |
| `hip_pos` | **`-1.0`** | 不存在 | G1 鼓励髋部保持默认(防止外八) |
| `waist_pos_dof29` | **`-1.0`** | 不存在 | G1 鼓励腰部不扭 |
| `arm_pos_dof29` | **`-0.5`** | 不存在 | G1 鼓励手臂自然下垂 |
| `contact` | `-1.0` | 不存在 | 双足接触鼓励 |
| `clearance` | `-0.05` | 不存在 | 抬脚高度 |
| `feet_distance` | `-0.1` | 不存在 | 双脚距离合适 |
| `feet_swing_height` | `0.0` | 不存在 | WMP-g1 关闭(WMP 原版是 -20) |
| `torques` | `-6e-7` | `-0.0001` | G1 关节力矩权重轻 |

**缺失的关键 reward**(WMP-g1 加的,cleanWMPg1 必须实现):
- `_reward_alive`:return 1.0
- `_reward_hip_pos`:`dof_pos[:, [1, 2, 7, 8]]` 的平方和(髋 yaw/roll)
- `_reward_waist_pos`:`dof_pos[:, 12:13]`(腰部)
- `_reward_arm_pos`:`dof_pos[:, 13:27]` 偏离 default 的平方和(双手臂)
- `_reward_ankle_pos`:`dof_pos[:, [4,5,10,11]]` 偏离 default(踝关节)
- `_reward_stand_normal`:身体 up vector 与世界 up 的对齐
- `_reward_not_fly`:双足至少一个接触地面才返回 1
- `_reward_upper_action_rate`:`actions[:, 13:]` 的差分(抑制手部抖动)

## 7. Camera(depth)

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `use_camera` | **`True`** | **`False`** | WMP-g1 默认开,cleanWMPg1 因为 driver 问题关掉 |
| `position` | `[0.047645, 0.000299, 0.46268]` | 同(在 cfg 里) | G1 头部前视相机 |
| `y_angle`(pitch down) | `[48, 48]` 度 | 同 | |
| `horizontal_fov` | `58` 度 | 同 | |
| `near_clip / far_clip` | `0 / 2` m | 同 | 视野 0~2m |
| `original / resized` | `(64, 64)` | 同 | |
| `update_interval` | `5` (每 5 个 policy step 更新一次) | 同 | |
| `buffer_len` | `2`(stack 2 帧) | 同 | |
| `dis_noise` | `0.0` | 同 | |
| `invert` | `True` | 同 | |

**结论**:camera 参数一致,只是 WMP-g1 默认开,cleanWMPg1 因为驱动 segfault 关掉。

## 8. Asset(URDF / 接触定义)

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| URDF | `g1_29dof_zy.urdf`(不存在) | `g1_29dof.urdf`(标准版) | cleanWMPg1 更稳 |
| `penalize_contacts_on` | `["hip", "knee", "shoulder_yaw", "elbow", "torso"]` | 同 | ✓ |
| `terminate_after_contacts_on` | `["pelvis", "waist", "shoulder_pitch", "knee"]` | 同 | ✓ |
| `body_name` | `"waist"` | 同 | ✓ |
| `self_collisions` | `1`(禁用自碰) | 同 | ✓ |
| `flip_visual_attachments` | `False` | 同 | ✓ |

**结论**:一致。

## 9. Robot 默认姿态(init_state)

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `pos` | `[0.0, 0.0, 0.80]` m | 同 | ✓ |
| `default_joint_angles`(29 个) | 全部相同 | 同 | ✓ |

**结论**:完全一致。

## 10. Normalization / 命令 scale

| 项 | WMP-g1 | cleanWMPg1 | 评价 |
|---|---|---|---|
| `clip_observations` | `30.` | 同 | ✓ |
| `clip_actions` | `5.0` | 同 | ✓ |
| `base_height` | `0.75` | `0.78` | WMP-g1 用 0.75(更接近实际 pelvis height 0.7429) |
| `obs_scales` | lin_vel 1.0 / ang_vel 0.25 / dof_pos 1.0 / dof_vel 0.05 | 同 | ✓ |

**结论**:`base_height` 改成 0.75 更准。

## 总结:cleanWMPg1 应该做的调整

### 必改(参考 WMP-g1)
1. **terrain**: `mesh_type='plane'`(简单起点)
2. **DR**:全部 False
3. **noise**: `add_noise=False`
4. **action_scale**: `0.5`
5. **ankle stiffness**: `100`,damping `5`
6. **commands**:lin_vel_x `[0.0, 0.8]`,ang_vel_yaw `[-1.0, 1.0]`
7. **reward weights**: tracking_lin_vel=20, tracking_ang_vel=20, base_height=-10, collision=-20, hip_pos=-1, waist_pos=-1, arm_pos=-0.5, action_rate=-5e-3
8. **base_height**: `0.75`

### 必须实现的 reward 函数
- `_reward_alive`(return 1.0)
- `_reward_hip_pos`(髋 yaw/roll 平方和)
- `_reward_waist_pos`(腰部)
- `_reward_arm_pos`(手臂偏离 default)
- `_reward_ankle_pos`(踝偏离 default)
- `_reward_stand_normal`(身体朝上)
- `_reward_not_fly`(双足接触)
- `_reward_upper_action_rate`(手臂动作差分)

### 可以保持现状
- Camera 参数(除了 use_camera)
- URDF / asset / contact 定义
- Default joint angles
- All observation scales
- PPO 算法参数

### Camera 决策
- 如果**环境能渲染 depth**(driver < 530 / CUDA 11):把 `use_camera=True` 恢复
- 如果**环境 segfault**(driver 580 / CUDA 13):保持 `False`,并在 README 标注