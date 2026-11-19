# biped_v1: tighter yaw tracking kernel from MAC-14, M5: drift shrinks, yaw control regresses (MAC-19, WP4-22)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 8f94394 (main), git_dirty false
train: biped_v1, --dr --track-yaw-std2 0.1, 256 envs, 1000 it (3000 -> 4000), 8 workers, seed 0,
       --resume mac14/model_3000.pt; 658 s
eval: eval_biped.py --dr --robustness --envs 8 --seconds 20; runtime_sim.py 20 s real time
      (no training running on the Mac during the runtime runs)
```

## Results (vx m/s / wz rad/s, 0 falls in every case)
| case | MAC-18 (MAC-14 policy) | **MAC-19** |
|---|---|---|
| 0.15, DR off | 0.178 / −0.186 | 0.153 / −0.046 |
| 0.25, DR off | 0.264 / −0.239 | 0.242 / +0.054 |
| 0.15, DR on | 0.150 / −0.128 | 0.136 / −0.169 |
| 0.25, DR on | 0.228 / −0.159 | 0.217 / −0.022 |
| zero command, DR off / on: wz | −0.021 / +0.001 | **−0.349 / −0.265** |
| turn 0.8, DR off / on: wz | 0.622 / 0.558 | **0.139 / 0.152** |
| runtime 0.15 | 0.194 / −0.190 (MAC-16) | 0.162 / −0.210 |
| runtime 0.25 | 0.285 / −0.148 (MAC-16) | 0.246 / −0.004 |

Latency 20 / 40 / 60 ms: vx 0.139 / 0.125 / 0.114 (93 / 83 / 76 %). Runtime: fell false,
latched 0, deadline_misses 0, 1000 periods, period_max 24.9 ms. MLP parity 1.3e-6.
Knee p95 0.19–0.26 N·m, max 0.52 (the clip) in several cases. `air_time` stayed around 0.029,
so the gait was kept. Reward/step 0.101 → 0.106.

## Conclusion
- Speed tracking got closer to the command (102 / 97 % with DR off), and the forward
  drift at 0.25 is gone (Python and runtime).
- **But yaw control regressed.** The policy now turns −0.27 to −0.35 rad/s on the spot
  at a zero command, and follows a 0.8 rad/s turn command at only ~0.15. At 0.15 forward,
  the DR-on and runtime drift is still −0.17 to −0.21.
- With the tighter kernel the yaw term pays little whatever the policy does
  (`track_angular_velocity` ~0.0025–0.0029 per step all run), so the 1000 iterations
  traded turning ability for the forward cases. **MAC-14 stays the reference policy.**
- Ideas (cloud): yaw curriculum (std² 0.5 → 0.1), oversampling zero-yaw and turn
  commands, or mirror-symmetry augmentation instead of a tighter kernel.

Files: `mac19/` holds `run.json`, `eval.json`, `report.json` (history every 10th iteration),
`runtime-v0.15.json`, `runtime-v0.25.json`, `mac19-mlp.json` (local paths shortened).
policy.onnx sha256 `a793cbb9b579bfdb5218ca4a467b86c9630ed721f7dcd6fe087ca60309213fca` (not committed).
