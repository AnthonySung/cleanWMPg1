# WMP-g1 vs WMP 差异对比分析报告

> **目的**:评估 `AnthonySung/WMP-g1` 相对原版 `bytedance/WMP`(基于 A1)的改动,识别哪些是 **G1 适配必需** 的、哪些是 **作者新增算法** 的、哪些是 **可能有 bug** 的,为 cleanWMPg1 的实现选型提供依据。

---

## 1. 总览

| 维度 | bytedance/WMP (原版) | AnthonySung/WMP-g1 | 类别 |
|---|---|---|---|
| 机器人 | A1 (12 DoF, 四足) | G1 (29 DoF, 人形) | 必需 |
| AMP 参考运动 | `datasets/mocap_motions/`(Laikago/A1 mocap) | `datasets/g1/`(19 个 humanoid csv) | 必需 |
| URDF | `resources/robots/a1/urdf/a1.urdf` | **未提交**,依赖外部 `resources/robots/g1_description/g1_29dof_zy.urdf` | 必需但仓库缺失 |
| world model | 同 | prop 维度 12(只保留 base/command/gravity) | 算法改动 |
| 训练任务 | `a1_amp` 单一 | `a1_amp` + `g1` + `g1_DK` 三套 | 必需 |
| AMP 开关 | 默认 True | 通过 `env.env.use_amp` 控制,G1 默认 False | 改动 |
| DK(Deep Koopman)模块 | 无 | 新增(独立 PPO + runner + actor_critic) | **作者新增算法** |
| 对称增强 (sym_coef) | 无 | 新增(用 e2cnn 的 EMLP,左右对称) | **作者新增算法** |
| 双修正机制 (h_t, a_t) | 无 | README 提及 "DK_ymloss 阶段完成性版本" | **作者新增算法** |
| 相机深度图渲染 | GPU tensor | 改回 CPU `get_camera_image` | 可疑改动 |
| reward / PD / action_scale | 单一 A1 调参 | G1 重新调参(高刚度、action_scale=0.5、关节限位奖励重做) | 必需 |

---

## 2. 逐文件差异详解

### 2.1 机器人配置(必需)
**`legged_gym/envs/g1/g1/g1_config.py`(WMP-g1 新增,505 行)**
- 相比 A1 的关键变更:
  - `num_dofs / action_dim = 29`(从 12)
  - 默认关节角度:双腿 12 + 腰部 3 + 双臂 7×2 = 29 个关节
  - 静止站立高度 `0.80`,`base_height_target=0.80`
  - 接触惩罚:`["hip", "knee", "shoulder_yaw", "elbow", "torso"]`(不包含 waist,因会自碰撞)
  - 终止接触:`["pelvis", "waist", "shoulder_pitch", "knee"]`
  - `forward_height_dim=525`(深度图预测范围)
  - 全身 PD:hip/knee 200,ankle 100,腰部 200,肩膀 20,肘 20,手腕 5/0.2
  - `action_scale=0.5`(A1 是 0.25,因为 G1 关节多、范围大)
  - `proprioception` prop 维度 = `3+3+3+3+29+29+29+2 = 101`(但 config 写的 `69 + 29 = 98`,少了 2 phase,config 不一致 ⚠️)
- 评估:**配置大体合理,但 prop 维度计算需要重新对齐 prop_dim 公式**

### 2.2 URDF 资源(必需但仓库缺失 ⚠️)
- WMP-g1 仓库 **没有** `resources/robots/g1_description/`
- config 写死路径 `'{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_zy.urdf'`
- **`g1_29dof_zy.urdf` 在公开仓库也找不到**,只有 `g1_29dof.urdf`(standard)和 `g1_29dof_rev_1_0.urdf`
- 这是个**隐患**,需要 fallback 到 `g1_29dof.urdf` 并验证关节顺序匹配

### 2.3 AMP 数据集(必需)
**`datasets/g1/*.csv`(19 个文件)** vs **`datasets/mocap_motions/`(Laikago)**

- A1 的 motion loader `AMPLoader`:
  - 硬编码 `JOINT_POS_SIZE = 12`
  - 列布局:`pos(3) + rot(4) + joint_pos(12) + tar_toe_pos(12) + lin_vel(3) + ang_vel(3) + joint_vel(12) + tar_toe_vel(12) = 61 维`
  - 是 **Laikago 风格** mocap
- G1 的 motion loader `AMPLoader_g1`:
  - 重写,支持任意 num_dofs(默认 29)
  - 显式 `set_data_index()`、`selected_joint_indices`
  - 转换 `pos(3)+rot(4)+joint_pos(29)+lin_vel(3)+ang_vel(3)+joint_vel(29)` 后再算
  - 但 G1 的 csv 不是 SMPL/SMPLX 的 mocap,而是 **LAFAN1 风格**(根位置+四元数+关节角度)
  - 检查点:CSV 文件需要进一步 inspect 列格式才能确认映射

**结论**:AMP 数据集迁移是核心工程,**绝不能简单 copy**,loader 必须重写。

### 2.4 actor_critic 模块
**`rsl_rl/modules/actor_critic_wmp.py`(原版)** vs **`rsl_rl/modules/actor_critic_DKwmp.py`(WMP-g1 新增)**

- `ActorCriticDKWMP` = `ActorCriticWMP` + `DeepKoopman` 模块,接在 actor 输入前做状态压缩
- 新增参数:`dk_latent_dim=128`, `num_history=5`, `use_observation_function`, `w_delta_s`, `w_delta_a`
- 新增损失:`ymloss` (双修正:预测 h_t 和 a_t)
- 历史:原版 `history_dim` 直接拼入 encoder;DK 版把历史拉给 DeepKoopman 做观测函数

**结论**:**作者新增算法**,不是 G1 适配必需。cleanWMPg1 应该**不包含**,保持与原版一致。

### 2.5 algorithm & runner
- `amp_ppo.py`(原版)→ `amp_ppo_DK.py`(DK 训练)+ `amp_ppo_sym.py`(对称增强)
- `wmp_runner.py`(原版)→ `wmp_runner_DK.py`(DK runner)+ `wmp_runner_g1.py`(G1 runner,核心改动是 `env_name` 分支)
- 共有的纯工程改动(必要):
  - `use_amp` 标志化,A1 默认 True,G1 默认 False(G1 无 AMP 可用时不启用判别器)
  - `update()` 拆为 `update_amp()` / `update()`(后者不再训练判别器)
  - `runner_class_name` 通过 cfg 切换(`WMPRunnerG1` / `WMPDKRunnerG1`)

**结论**:
- `use_amp` 改造是**必要**(G1 训练可以无 AMP)
- DK / sym 都是**作者新增算法**

### 2.6 dreamer/networks.py(WM 部分)
- WMP-g1 几乎没改,只有几行注释打印
- `prop` 改成 12 维(README 第 14 行)的改动**只存在于 README 描述,没在 networks.py 找到对应改动**(可能漏改 ⚠️)
- world model 输入维度需要根据 G1 的实际 prop_dim 重算

### 2.7 camera 渲染(可疑改动 ⚠️)
```python
# 原版
depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, ...)
depth_image = gymtorch.wrap_tensor(depth_image_)
# WMP-g1
depth_image_ = self.gym.get_camera_image(self.sim, ...)  # CPU 版本
depth_image = torch.tensor(depth_image_, device=self.device)
```
- 改成 CPU 渲染后 `to(device)`,**必然成为训练瓶颈**
- 推测原因:Isaac Gym 版本兼容性(Preview 3 可能在某些环境下 GPU tensor 接口有问题)
- **cleanWMPg1 应该用 GPU 版本,作为默认**

### 2.8 play.py / train.py
- `play.py`:新增 `RENDER=True` 时附加一个外置相机录视频(与训练用的 depth camera 无关)
- `train.py`:`env_name == 'g1'` 时切换到 `make_wmp_runner_g1`
- 这部分是**必需**的工程化改造

### 2.9 辅助文件
- `setup.py`、`legged_gym.egg-info`、`rsl_rl.egg-info`:正常 pip install 产物
- `symm_utils.py`:e2cnn 对称增强工具(作者新增)
- `state_estimator_DK.py`:DeepKoopman 模型本体(作者新增)
- `g1_config (2).py / g1_config copy.py / g1_config_origin.py`:作者的迭代版本,只需要最新一份

---

## 3. WMP-g1 仓库中**疑似 bug / 风险点**

| # | 位置 | 问题 | 严重性 |
|---|---|---|---|
| 1 | `g1_config.py` | `prop_dim = 69 + 29 = 98`,但实际 prop 应该是 `3+3+3+3+29+29+29+2 = 101`,公式与字段不匹配 | 高 |
| 2 | `g1_config.py` | URDF 文件名 `g1_29dof_zy.urdf` 在公开渠道找不到,需 fallback | 高 |
| 3 | README 7.17 行 | 提到 WM 的 prop 改成 12 维,代码里看不到对应改动 | 中 |
| 4 | `legged_robot.py` | depth image 改 CPU 渲染,会拖慢训练 | 中 |
| 5 | `g1_envs/` 与 `g1/` | 存在两份重复的 g1 任务定义(`legged_robot_g1.py`),可能导入混乱 | 中 |
| 6 | `g1_config_dof27.py` 等 | 多份历史 config,难以判断哪个是 ground truth | 低 |
| 7 | `amp_ppo_sym.py` | 用 e2cnn 引入对称增强,但 e2cnn 安装繁琐,且 README 没提论文出处 | 中 |
| 8 | `actor_critic_DKwmp.py` 注释 `# TODO: 上层传参 num_history, prop_dim` | 顶层 TODO 未完成 | 高(作者自己也认为没做完) |

---

## 4. cleanWMPg1 设计原则

1. **只做"机器人切换 + 必要工程化",不做任何算法创新**
   - ✅ 改 config:`a1` → `g1`,关节数 12 → 29,PD / 静止姿态 / 终止接触
   - ✅ 改 env 注册:G1 env 类 + G1 config 类
   - ✅ 改 AMP loader:`AMPLoader_g1`,支持 29 DoF
   - ✅ 改 wmp_runner:`use_amp` 标志化,`env_name` 分支
   - ✅ 补 URDF:从 unitree_rl_gym 复制 g1_29dof 系列,创建 resources/robots/g1_description/
   - ❌ 不引入 DeepKoopman(`amp_ppo_DK.py` / `state_estimator_DK.py` / `actor_critic_DKwmp.py` / `wmp_runner_DK.py`)
   - ❌ 不引入对称增强(`amp_ppo_sym.py` / `symm_utils.py`)
   - ❌ 不引入双修正 ymloss
2. **结构与原版一致**,只是 `a1` 路径变成 `g1`,便于后续对照论文
3. **不加 author 私货**:不引入 `g1_envs`(重复目录)、不留 `g1_config (2).py` 之类的迭代版本
4. **奖励 / 域随机化参数**:从 WMP-g1 借鉴(已调好),但保留原版文件名

---

## 5. cleanWMPg1 文件清单(目标)

```
cleanWMPg1/
├── README.md                       # 改:加 G1 训练命令
├── requirements.txt                # 复制
├── LICENSE.txt, licenses/          # 复制
├── cmd_lines_g1.txt                # 新:命令参考
├── ANALYSIS.md                     # 本文件
├── legged_gym/
│   ├── setup.py                    # 复制
│   ├── envs/
│   │   ├── __init__.py             # 改:注册 G1 任务
│   │   ├── base/                   # 复制原版(不加调试 print)
│   │   ├── g1/                     # 新:替换 a1
│   │   │   ├── base/               # 复制 a1 对应文件改名为 g1
│   │   │   └── g1/
│   │   │       └── g1_amp_config.py
│   ├── scripts/
│   │   ├── train.py                # 改:支持 --task=g1_amp
│   │   └── play.py                 # 改:G1 分支
│   └── resources/robots/
│       └── g1_description/         # 新:从 unitree_rl_gym 复制 URDF
├── rsl_rl/
│   ├── setup.py                    # 复制
│   ├── algorithms/
│   │   ├── amp_ppo.py              # 改:支持 use_amp 标志
│   │   └── amp_discriminator.py    # 复制
│   ├── modules/
│   │   └── actor_critic_wmp.py     # 改:支持 G1 维度,移除 DK
│   ├── runners/
│   │   ├── wmp_runner.py           # 改:use_amp + env_name 分支
│   │   └── wmp_runner_g1.py        # 可选:若 runner 也需要重命名
│   ├── datasets/
│   │   ├── motion_loader.py        # 复制
│   │   └── g1_motion_loader.py     # 新:AMP G1 数据加载
│   └── storage/rollout_storage.py  # 复制
├── dreamer/                        # 复制原版,确认 prop_dim 与 G1 一致
└── datasets/
    ├── mocap_motions/              # 保留(A1 训练可选)
    └── g1/                         # 复制
```

---

## 6. 验证清单

迁移完成后,以下检查必须通过才算 clean:

- [ ] `python legged_gym/scripts/train.py --task=g1_amp --headless` 能启动(env 数 4096 起)
- [ ] 没有 DK 模块的导入(grep `DeepKoopman` 应为空)
- [ ] 没有 e2cnn 依赖
- [ ] 4096 个 env 不爆显存(RTX 3090 24G 目标)
- [ ] `prop_dim + privileged_dim + height_dim` 与 actor_critic 输入维度对得上
- [ ] AMP loader 能正确读取 `datasets/g1/walk.csv`,打印前几行确认列数
- [ ] play.py 能加载 checkpoint 并跑通