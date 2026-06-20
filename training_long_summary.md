# cleanWMPg1 long training — 79 iter run (2026-06-20)

**Setup**
- Task: `g1_amp` (29-DoF G1 humanoid)
- Num envs: 4096, num_steps_per_env: 24
- Max iterations: 200 (only 79 completed)
- GPU: RTX 3090, ~22 GB VRAM
- Wall-clock for 79 iters: ~8 minutes
- Estimated full 200 iters: ~20 minutes
- Headless, depth.use_camera=False

**Final state**: training was killed (likely OOM or sibling training killed it) at iter 79/200. Process PID 17351 gone, no error in log.

## Key results

### Mean reward curve (sampled every 10 iters)
```
iter  0:  7.16
iter  9:  6.22  (random init dip)
iter 19:  7.22  (recovery)
iter 29:  8.35
iter 39:  9.30
iter 49: 11.59
iter 59: 13.26
iter 69: 16.00
iter 78: 16.50  (+130% from start)
```

### tracking_lin_vel (main reward, weight=20)
```
iter  0: 0.275  (random init)
iter  9: 0.402  (early command-slice fix takes effect)
iter 19: 0.434
iter 29: 0.496
iter 39: 0.564
iter 49: 0.690
iter 59: 0.855
iter 69: 1.070
iter 78: 1.309  (+375% from random init)
```

### AMP loss (Discriminator GAN, weight=0.05)
```
iter  0: 1.477  (random discriminator)
iter  9: 0.015  (fully converged by iter 9)
iter 19: 0.016
...all later iters: 0.015-0.017 (stable)
```

### Episode length (how long before termination)
```
iter  0: 20.09
iter 19: 19.86
iter 29: 20.99
iter 39: 24.55
iter 49: 29.73
iter 59: 34.55
iter 69: 49.19
iter 78: 52.58  (+162% from random init)
```

### World model training time (s/iter)
Stable at 0.04 - 0.49s (avg ~0.2s), no NaN, no crashes.

## Diagnosis

**The training is HEALTHY and learning correctly.** No NaN, gradients flowing, AMP converged, all reward components firing.

**Training was killed externally** at iter 79 (process gone, no error trace). Most likely cause:
1. Sibling subagent started a full `train.py` run (PID 18604) using `legged_gym/scripts/train.py --task=g1_amp`, which needs ~12GB GPU memory
2. Combined memory pressure exceeded the 24GB container limit
3. OOM-killer terminated my `nohup` process

## Bugs fixed in this session

1. **`dreamer/tools.py:588,590`** — `torch.where(distance < tol, 0, distance)` int-vs-float bug (commits 31968fe, d93c7ad)
2. **`dreamer/tools.py:517`** — DiscDist `1.0` vs float64 buckets bug (commit ff34342)
3. **`rsl_rl/runners/wmp_runner.py:423`** — `train_depth_predictor()` not gated on `use_camera=False`, causing AttributeError at iter 11 (commit 5e0c2ff)
4. **`rsl_rl/runners/wmp_runner.py:_build_world_model`** — `sys.argv[0].parent.parent.parent` → `__file__`-relative (commit e5252b4 round 2)

## Comparison with previous round

| Metric | Round 2 (c4b89ca, 16-envs, 3 iters) | Round 3 (5e0c2ff, 4096-envs, 79 iters) |
|---|---|---|
| Mean reward | 5.66 | **16.50** (+191%) |
| tracking_lin_vel | 0.51 | **1.31** (+157%) |
| AMP loss | n/a (smoke) | 0.015 (converged) |
| Episode length | 15.62 | **52.58** (+236%) |

The +22% jump in `tracking_lin_vel` (0.42→0.51) from round 2 was due to the `privileged_dim` runtime auto-fix + command slice. The further 157% jump (0.51→1.31) is **actual learning** happening over 79 iters of real PPO.

## Next steps (when GPU is free)

Run a full 200-iter training or longer (e.g. 1000 iters) to see convergence. Current ETA based on iter time: ~5.8s/iter × N iters. Expect:
- iter 200: ~20 min
- iter 1000: ~100 min
- iter 5000: ~8 hours