# biped_v1: milder yaw kernel (std² 0.25) from MAC-14, M5: drift halved, not accepted (MAC-21, WP4-22)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 5633e4d (main), git_dirty false
train: biped_v1, --dr --track-yaw-std2 0.25, 256 envs, 1000 it (3000 -> 4000), 8 workers, seed 0,
       --resume mac14/model_3000.pt; 444 s
eval: eval_biped.py --dr --robustness --envs 8 --seconds 20; runtime_sim.py 20 s real time
      (no training running during the runtime runs)
```

## Results (vx m/s / wz rad/s; 0 falls in every case)
| case | MAC-14 | MAC-19 (std² 0.1) | **MAC-21 (std² 0.25)** |
|---|---|---|---|
| 0.15, DR off | 0.178 / −0.186 | 0.153 / −0.046 | 0.170 / +0.103 |
| 0.25, DR off | 0.264 / −0.239 | 0.242 / +0.054 | 0.260 / +0.097 |
| 0.15, DR on | 0.150 / −0.128 | 0.136 / −0.169 | 0.142 / +0.044 |
| 0.25, DR on | 0.228 / −0.159 | 0.217 / −0.022 | 0.222 / +0.040 |
| zero command wz, off / on | −0.021 / +0.001 | −0.349 / −0.265 | +0.047 / +0.035 |
| turn 0.8 wz, off / on | 0.622 / 0.558 | 0.139 / 0.152 | 0.625 / 0.536 |
| latency 40 ms vx (of 0.15) | 86 % | 83 % | 81 % |
| runtime 0.15 | 0.194 / −0.190 | 0.162 / −0.210 | 0.185 / +0.165 |
| runtime 0.25 | 0.285 / −0.148 | 0.246 / −0.004 | 0.276 / +0.080 |

Runtime: fell false, latched 0, deadline_misses 0, 1000 periods, period_max 28.1 ms.
MLP parity 1.4e-6. Latency 20 / 60 ms: 95 / 73 %. friction_0.5 shows wz +0.205.
Knee p95 0.17–0.22 N·m, max at the 0.52 clip in several DR-on cases.

## Verdict against the MAC-21 acceptance rows
| row | MAC-21 | pass? | MAC-14 |
|---|---|---|---|
| \|wz\| ≤ 0.08 at 0.15 / 0.25, DR off | 0.103 / 0.097 | **no** | 0.186 / 0.239 |
| \|wz\| ≤ 0.08 at 0.15 / 0.25, DR on | 0.044 / 0.040 | yes | 0.128 / 0.159 |
| \|wz\| ≤ 0.05 at zero command | 0.047 / 0.035 | yes | 0.021 / 0.001 |
| turn 0.8 → wz ≥ 0.5 | 0.625 / 0.536 | yes | 0.622 / 0.558 |
| 40 ms latency ≥ 80 % of vx | 81 % | yes | 86 % |
| 0 falls | 0 | yes | 0 |

**Not accepted.** MAC-21 roughly halves the walking drift and keeps the ability to turn
(unlike MAC-19), but the drift has flipped to the left, and with DR off it is still
above 0.08. The zero-command, turn (DR on) and latency rows are slightly worse than
MAC-14. **MAC-14 stays the reference.** MAC-21 is the best yaw candidate so far. The runtime at
0.15 still yaws +0.165.

Files: `mac21/` holds `run.json`, `eval.json`, `report.json` (history every 10th iteration),
`runtime-v0.15.json`, `runtime-v0.25.json`, `mac21-mlp.json` (local paths shortened).
policy.onnx sha256 `1e4dcd5c26e7bfed9f7f9704e1632b8b9f047f96c2bc4cd903e0538f6581eb02` (not committed).
