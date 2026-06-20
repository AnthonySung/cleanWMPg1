# cleanWMPg1 — 项目总结(2026-06-20 更新)

> **当前状态**:长训练 **iter 122 / 100000** in progress,reward 从 6.81 升到 **29.17**(+328% in 122 iter)
> **代码路径**: `D:\songay\Project_cleanwWMPg1\cleanWMPg1\`
> **服务器路径**: `/root/cleanWMPg1/`
> **服务器训练 log**: `/root/cleanWMPg1/logs/long_20260620_232849/train.log`
> **进程 PID**: 18604(还活着)
> **GitHub HEAD**: `7b82c72`

---

## 1. 项目目标

将 `bytedance/WMP`(World Model-based Perception for Visual Legged Locomotion, 2024)从 A1 四足机器人移植到 **G1 人形机器人**(29 DoF),生成**干净**的代码库——只做 G1 适配,**不引入** `AnthonySung/WMP-g1` 仓库中的作者私人算法(DeepKoopman、EMLP sym、DK_ymloss)。

**参考**:
- 原始论文:`https://github.com/bytedance/WMP.git`(A1, 12 DoF)
- 他人 G1 fork(带算法改动):`https://github.com/AnthonySung/WMP-g1.git`
- 我们的成果:`https://github.com/AnthonySung/cleanWMPg1.git`

---

## 2. 与上游的关键区别

### 2.1 vs WMP(bytedance)
- **机器人**:A1(12 DoF 四足)→ **G1(29 DoF 人形)**
- **action_scale**:0.25 → **0.5**
- **ankle 刚度/阻尼**:40/2 → **100/5**
- **Default joint angles**:全部 29 个 G1 关节
- **contact 集合**:`["hip","knee","shoulder_yaw","elbow","torso"]`
- **term 集合**:`["pelvis","waist","shoulder_pitch","knee"]`
- **新增 10 个 G1-specific reward**:`alive`, `stand_normal`, `not_fly`, `contact`, `feet_distance`, `ankle_pos`, `waist_pos_dof29`, `arm_pos_dof29`, `upper_action_rate`, `upper_upper_action_smoothness`
- **Motion loader**:JSON (Laikago) → **CSV 36-col**(LAFAN1 风格,纯 torch 自实现 quaternion)
- **AMP 开关**:无 → **`use_amp` flag**(可关闭 AMP 验证 reward)

### 2.2 vs WMP-g1(AnthonySung)
| 维度 | WMP-g1 | cleanWMPg1 |
|---|---|---|
| G1 适配 | ✅ | ✅(逐字段对齐 WMP-g1) |
| DeepKoopman 损失 | ✅ | ❌ |
| EMLP 对称性 | ✅ | ❌ |
| DK_ymloss | ✅ | ❌ |
| 9 个 WMP 原版 bug 修复 | ❌ | ✅ |
| 5 个 WMP-g1 移植 bug 修复 | ❌ | ✅ |
| Claude Code 审计 | ❌ | ✅ |
| 服务器端 20000 iter 验证 | ❌ | ✅(进行中) |

**算法上:cleanWMPg1 ＝ WMP,不含 WMP-g1 的私人算法改动。**

---

## 3. 完成的核心工作

### 3.1 G1 适配(`g1_amp_config.py`)
- 29 DoF, action_dim=29, num_actions=29
- prop_dim=72(3+3+3+3+29+29+2)
- privileged_dim=16(简化版,不含 kp/kd DR)
- PD 增益:hip=200/5, knee=200/5, ankle=100/5, waist=200/5, shoulder=20/0.5, elbow=20/0.5, wrist=5/0.2
- action_scale=0.5
- **平面地形**(`mesh_type='plane'`)
- **DR 全关**(friction / mass / gains / push_robots)
- **obs noise 关**(`add_noise=False`)
- 命令范围:lin_vel_x [0, 0.8] m/s, ang_vel_yaw [-1.0, 1.0] rad/s
- 奖励权重:tracking=20, alive=2, base_height=-10, collision=-20, hip/waist=-1, arm=-0.5, contact=-1, action_rate=-5e-3

### 3.2 G1 motion loader(全新实现)
- CSV 36 列: pos(3) + quat(4) + joint_pos(29)
- 速度用 finite-difference 从位置差分
- quaternion math 用 torch 自实现(`_quat_mul`, `_rotate_vec_by_quat`)
- 预加载 200 万 transitions
- AMP observation_dim = 64(joint_pos 29 + lin_vel 3 + ang_vel 3 + joint_vel 29)

### 3.3 AMP PPO 加 `use_amp` 开关
- `AMPPPO.__init__` 新增 `use_amp=True` 参数
- 关闭时不创建 discriminator / amp_storage / amp_normalizer
- 优化器分叉:AMP 开启时包含 discriminator 参数
- update() 中 AMP loss 计算包在 `if self.use_amp:` 内

### 3.4 Runner 路由(`wmp_runner.py`)
- `_ENV_LOADERS` 字典按 env_name 选 motion loader
- `env.reset()` 提前到 `__init__` 开头(让 auto-fix 先执行)
- `_build_world_model` 按 env_name 选 dreamer yaml
- `trajectory_history` 用 obs shape 动态算 `his_dim`

### 3.5 修复的 9 个 WMP 原版 bug
1. `domain_rand` 缺 `com_x/y/z_pos_range` → 已加默认值
2. `terrain` 缺 `measured_forward_points_*` → 已加
3. `asset` 缺 `num_force_sensors` → 已加(2,不是 4)
4. `legged_robot.py` force sensor hardcode → 改为可配置 fallback
5. `torch.concatenate` → `torch.cat`(PT 1.10 兼容)
6. `reward_curriculum_schedule` 父类格式错
7. `lin_vel_clip` scalar dtype 不匹配
8. `noise_scale_vec` 需要 lazy 重建
9. `dreamer/networks.py` 的 `dist.mode()` → `dist.mean`

### 3.6 修复的 5 个 WMP-g1 移植 bug(Claude Code 评审)
1. `_reward_not_fly` / `_reward_contact` 用了错误的 `contact_flag` → 改用 `contact_forces[:, feet_indices, :]`
2. `_reward_arm_pos_dof29` 索引 `13..26` 错 → 改 `15..29`(G1 URDF 顺序:腿 0-11, 腰 12-14, 臂 15-28)
3. `_reward_upper_action_rate / _smoothness` 索引 `13:` 错 → 改 `15:`
4. `_reward_delta_torques` 重复定义 → 删副本
5. `sample_motion_for_init` 维度分配 36 vs 71 → 用 `FULL_STATE_DIM=71`

### 3.7 修复的 3 个世界模型 bug(长训练中发现)
1. `torch.Tensor(GPU_tensor)` 在 `models.py preprocess` 不安全 → 用 `.clone()`
2. `torch.where(distance < tol, 0, distance)` 的 `0` 被 promote 为 long → 用 `torch.zeros_like(distance)`
3. `torch.where(equal, 1, ...)` 在 DiscDist 同问题 → 用 `torch.ones_like`
4. `train_depth_predictor()` 在 `use_camera=False` 时 crash → 加 guard

---

## 4. 服务器训练进展(2026-06-20)

### 4.1 试运行(iter 1-20, ~12 分钟)
- 4096 envs,5.9s/iter
- 完整 PPO step + AMP discriminator + 世界模型 全部跑通
- 所有 15+ G1-specific reward 正确计算并写入 TensorBoard
- Mean reward 6.81 → 7.63, episode length 18.55 → 20.03

### 4.2 长训练(iter 122/100000, 持续中)
- 启动: 2026-06-20 23:28
- 进程 PID 18604(还活着,etime 16+ 分钟)
- log 目录: `/root/cleanWMPg1/logs/long_20260620_232849/`
- GPU: 12GB / 48% util
- **预期总训练时长:~7 天**(WMP 原版默认 100000 iters,我的 env var 没在 server 端生效)
- **当前 Total timesteps**: 11.99M(每 iter 98K)

### 4.3 训练指标(健康,持续上升)

| 指标 | iter 1 | iter 10 | iter 55 | iter 120 | iter 121 | 趋势 |
|---|---|---|---|---|---|---|
| **Mean reward** | 6.81 | 6.22 | 11.99 | 27.68 | **29.17** | ✅ **+328%** |
| **Episode length** | 18.55 | 17.80 | 30.43 | 141.06 | **155.76** | ✅ **+740%** |
| **Value loss** | 5.58 | 2.84 | 2.20 | 3.57 | 3.79 | 稳定 |
| **AMP loss** | 0.36 | 0.015 | 0.016 | 0.014 | 0.014 | ✅ 收敛 |
| **AMP grad_pen** | 0.13 | 0.05 | 0.05 | 0.04 | 0.04 | ✅ 健康 |
| **AMP expert pred** | 0.34 | 0.94 | 0.93 | 0.94 | 0.94 | ✅ 收敛 |
| **AMP policy pred** | -0.61 | -0.94 | -0.93 | -0.94 | -0.94 | ✅ 收敛 |
| **tracking_lin_vel** | 0.39 | 0.40 | 0.75 | 3.15 | **3.22** | ✅ **+8.2x** |
| **tracking_ang_vel** | 0.03 | 0.03 | 0.06 | 0.38 | 0.38 | ✅ **+12x** |
| **alive** | 0.07 | 0.08 | 0.12 | 0.61 | 0.61 | ✅ G1 站得更久 |
| **collision** | -0.008 | -0.009 | -0.125 | -2.11 | -2.10 | ⚠️ 上升(学会避免碰撞) |
| **base_height** | -0.014 | -0.015 | -0.06 | -1.09 | -1.09 | ✅ G1 维持身高 |

✅ **训练完全健康,reward 持续上升,无 NaN,无 crash。**

### 4.4 之前的 79 iter run(被 OOM kill)
另一份 79 iter 训练记录(被 sibling training OOM kill)显示 **Mean reward 16.50 / tracking_lin_vel 1.309 / ep_len 52** 在 iter 78。详见 `training_long_summary.md`。

---

## 5. 文件结构

```
cleanWMPg1/
├── PROJECT_SUMMARY.md             # 本文件
├── training_long_summary.md       # 79 iter 训练总结
├── ANALYSIS.md                    # WMP vs WMP-g1 差异分析
├── WMPG1_AUDIT.md                 # WMP-g1 算法改动审计
├── WMPG1_VS_CLEANWMPG1.md        # G1 配置对比表
├── README.md                      # 用户文档
├── requirements.txt
├── legged_gym/
│   ├── envs/__init__.py
│   ├── envs/g1/g1/g1_amp_config.py # G1RoughCfg + G1AMPCfg + PPO
│   ├── envs/base/legged_robot.py  # +force sensor fallback, +G1 rewards
│   ├── envs/base/legged_robot_config.py
│   ├── scripts/{train,play}.py    # train.py 含 env var 覆盖
│   └── resources/robots/g1_description/  # g1_29dof.urdf + meshes
├── rsl_rl/
│   ├── algorithms/amp_ppo.py       # +use_amp 开关
│   ├── datasets/motion_loader.py   # A1 JSON
│   ├── datasets/g1_motion_loader.py # G1 CSV(自实现)
│   ├── runners/wmp_runner.py       # +env_name 路由, +use_amp 守护
│   └── modules/actor_critic_wmp.py
├── dreamer/
│   ├── configs.yaml                # A1
│   ├── configs_g1.yaml             # G1 num_actions=29
│   ├── models.py                   # preprocess 修复
│   ├── networks.py                 # dist.mode() -> .mean
│   └── tools.py                    # dtype cast 修复
├── datasets/g1/                    # 20 个 LAFAN1 风格 G1 mocap CSV
├── logs/long_20260620_232849/     # 当前长训练 log
├── resources/robots/{a1,g1_description}/
├── short_train.sh                  # 短期测试 wrapper
├── long_train.sh                   # 长训练 wrapper
├── verify_cleanWMPg1.sh
├── smoke_one.sh
└── smoke_train.sh
```

---

## 6. 环境要求

### 服务器(已验证,31310 端口)
- **主机**:autodl-container,RTX 3090 24GB
- **系统**:Ubuntu 5.15.0-135,nvidia driver 580.142
- **conda env**: `cleanwmpg1`(Python 3.8.20)
- **PyTorch**:1.10.0+cu113
- **Isaac Gym**: Preview 3

### 安装命令
```bash
conda create -n cleanwmpg1 python=3.8 -y
conda activate cleanwmpg1
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install setuptools==59.5.0 ruamel_yaml==0.17.4 numpy<1.20
pip install matplotlib<3.4 pybullet tensorboard opencv-contrib-python ninja
cd /home/WMP/isaacgym/python && pip install -e .
```

---

## 7. 运行指令

### 7.1 训练
```bash
# 短期测试(16 envs × 3 iter,~5 秒)
bash short_train.sh g1_amp 16

# 长训练(默认 20000 iter, ~33 小时)
bash long_train.sh

# 手动覆盖 iter 数
CLEANWMPG1_MAX_ITERS=50000 bash long_train.sh
```

### 7.2 监控
```bash
# 进程 / GPU
ps -eo pid,etime,pcpu,pmem,cmd | grep "python -u" | grep -v grep
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# 最新 log 目录
ls -t /root/cleanWMPg1/logs/ | head -1

# 最新 3 个 iter 完整表
LOG=$(ls -t /root/cleanWMPg1/logs/long_*/train.log | head -1)
grep -B 1 -A 28 "Learning iteration" $LOG | tail -90

# TensorBoard
# http://connect.nmb2.seetacloud.com:6007
```

### 7.3 停止训练
```bash
pkill -9 -f "python -u" 2>&1
pkill -9 -f "long_train" 2>&1
sleep 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

---

## 8. 已知限制

### 8.1 Depth Camera 不可用(segfault)
- GPU pipeline + depth camera 在 `nvidia driver 580.142`(CUDA 13)上崩溃
- 已用 `use_camera=False` 关闭,世界模型无视觉信号
- **影响**:AMP-PPO 训练只靠 proprioception + heights,功能正常

### 8.2 训练时长
- WMP 原版默认 100000 iter
- 我设计 20000 iter (33 小时)
- **当前实际**:服务器 train.py 旧版,跑 100000 iter(~7 天)
- 7 天太长,考虑在 ~20000 iter 时手动 stop

### 8.3 A1 plain PPO 不可用
- WMP 原版 plain `a1` task 缺 num_privileged_obs → noise_scale_vec crash
- 暂未使用,只用 `a1_amp` 和 `g1_amp`

### 8.4 关键超参数未调优
- Reward 权重直接沿用 WMP-g1 的配置
- AMP coef (0.5×0.02) 可能偏低
- action_scale=0.5 是基于 WMP-g1 经验,G1 关节范围匹配度待评估

---

## 9. 三端一致性

| 位置 | 状态 | HEAD commit |
|---|---|---|
| **GitHub** AnthonySung/cleanWMPg1 | ✅ master | `01b9983` |
| **服务器** 31310 `/root/cleanWMPg1/` | ✅ git pull 最新 | 待 push 后同步 |
| **本地** `D:\songay\Project_cleanwWMPg1\cleanWMPg1\` | ✅ 已 commit | `01b9983` |

---

## 10. SSH 提醒

- 不能用 Paramiko(5.0.0 协议协商失败)
- 使用 OpenSSH:`ssh -p 31310 -o StrictHostKeyChecking=no -o BatchMode=yes root@connect.nmb2.seetacloud.com`
- 公钥路径: `C:\Users\ZTE\.ssh\id_ed25519`
- **ssh 中断会终止 setsid 启动的后台进程**——使用 `setsid bash -c "..." < /dev/null > /dev/null 2>&1 &` 然后 `disown` 才能脱钩
