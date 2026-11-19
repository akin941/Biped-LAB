# biped_v1 with mirror-symmetry augmentation from MAC-14, M5: training degrades (MAC-23, WP4-23)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 4329183 (main), git_dirty false
train: biped_v1, --dr --track-yaw-std2 0.25 --symmetry aug, 256 envs, 1000 it (3000 -> 4000),
       8 workers, seed 0, --resume mac14/model_3000.pt; 1221 s (2.75x MAC-21's 444 s)
eval: eval_biped.py --dr --robustness --envs 8 --seconds 20; runtime_sim.py 20 s real time
```

## Training trend (100-iteration means)
| it | reward/step | falls/it | knee torque mean (N·m) | track_angular_velocity |
|---|---|---|---|---|
| 0 | 0.098 | 1.14 | 0.20 | 0.0032 |
| 200 | 0.098 | 0.37 | 0.23 | 0.0025 |
| 400 | 0.090 | 1.06 | 0.24 | 0.0016 |
| 600 | 0.082 | 1.46 | 0.27 | 0.0011 |
| 900 | 0.068 | ~2.2 | 0.31 | 0.0008 |

After iteration ~100 the run gets steadily worse: reward falls, falls rise, and knee torque climbs towards the clip.

## Results (vx m/s / wz rad/s; 0 falls in every eval case)
| case | MAC-14 | MAC-21 | **MAC-23** |
|---|---|---|---|
| 0.15 / 0.25, DR off | −0.186 / −0.239 (wz) | +0.103 / +0.097 | 0.178 / **+0.164**, 0.251 / **+0.188** |
| 0.15 / 0.25, DR on | −0.128 / −0.159 | +0.044 / +0.040 | 0.135 / −0.026, 0.203 / −0.037 |
| zero command wz, off / on | −0.021 / +0.001 | +0.047 / +0.035 | +0.005 / +0.007 |
| turn 0.8 wz, off / on | 0.622 / 0.558 | 0.625 / 0.536 | **0.195 / 0.196** |
| latency 40 ms (of 0.15) | 86 % | 81 % | 79 % |
| runtime 0.15 / 0.25 (vx / wz) | 0.194 / −0.190, 0.285 / −0.148 | 0.185 / +0.165, 0.276 / +0.080 | 0.196 / **+0.446**, 0.265 / **+0.622** |
| knee p95 (DR on, 0.15) | 0.19 | 0.18 | **0.52 (clip)** |

Through the runtime the robot walks in circles: net displacement is 0.58 / 0.65 m in 20 s.
Runtime safety was clean (fell false, latched 0, deadline_misses 0, 1000 periods). MLP parity 1.4e-6.

## Verdict
Fails 3 of 6 rows: DR-off |wz|, turn 0.8, and 40 ms latency (79 %). It passes DR-on |wz|, zero
command and falls. **Worse than MAC-21 and MAC-14. MAC-14 stays the reference.**

The run was still getting worse when it stopped, so more iterations won't fix it.
Before another symmetry run, the cloud should check:
- the mirror maps for the action and the command (yaw and lateral sign), and the history of
  the latency buffer if it is part of the observation,
- PPO settings with the doubled batch (learning rate / KL, mini-batch size),
- why the knee torque climbs to the clip (in MAC-14 the p95 was about 0.19).

Files: `mac23/` holds `run.json`, `eval.json`, `report.json` (history every 10th iteration),
`runtime-v0.15.json`, `runtime-v0.25.json`, `mac23-mlp.json` (local paths shortened).
policy.onnx sha256 `6c5c3758e618736c81b4f4047a3629b21beee12d97ecf46adcb286c97ea27b6a` (not committed).
