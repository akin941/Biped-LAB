# biped_v1 with a mirror loss from MAC-14, M5: passes every row, new reference policy (MAC-25, WP4-24)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: f5f71eb (main), git_dirty false
train: biped_v1, --dr --track-yaw-std2 0.25 --symmetry loss --mirror-loss-coeff 0.5, 256 envs,
       1000 it (3000 -> 4000), 8 workers, seed 0, --resume mac14/model_3000.pt; 983 s
eval: eval_biped.py --dr --robustness --envs 8 --seconds 20; runtime_sim.py 20 s real time
```

## Training (100-iteration means)
| it | loss_symmetry | falls/it | reward/step | knee torque mean (N·m) |
|---|---|---|---|---|
| 0 | 0.0301 | 2.38 | 0.093 | 0.236 |
| 100 | 0.0093 | 0.25 | 0.106 | 0.204 |
| 300 | 0.0045 | 0.14 | 0.108 | 0.186 |
| 600 | 0.0046 | 0.08 | 0.110 | 0.175 |
| 900 | 0.0037 | 0.08 | 0.111 | 0.168 |

The symmetry loss falls by about 8×. Falls settle at 0.08–0.14 per iteration. The first 100
iterations show the same resume transient as MAC-21 (2.48), and the watchdog (from it 200,
100-iteration mean > 1.0) never fired. Knee torque goes down, where in MAC-23 it went up.

## Results (vx m/s / wz rad/s; 0 falls in every case)
| case | MAC-14 | MAC-21 | MAC-23 | **MAC-25** |
|---|---|---|---|---|
| 0.15, DR off | 0.178 / −0.186 | 0.170 / +0.103 | 0.178 / +0.164 | 0.166 / **−0.047** |
| 0.25, DR off | 0.264 / −0.239 | 0.260 / +0.097 | 0.251 / +0.188 | 0.252 / **+0.006** |
| 0.15, DR on | 0.150 / −0.128 | 0.142 / +0.044 | 0.135 / −0.026 | 0.142 / −0.052 |
| 0.25, DR on | 0.228 / −0.159 | 0.222 / +0.040 | 0.203 / −0.037 | 0.219 / −0.028 |
| zero command wz, off / on | −0.021 / +0.001 | +0.047 / +0.035 | +0.005 / +0.007 | −0.034 / −0.020 |
| turn 0.8 wz, off / on | 0.622 / 0.558 | 0.625 / 0.536 | 0.195 / 0.196 | 0.579 / 0.523 |
| latency 40 ms (of 0.15) | 86 % | 81 % | 79 % | 83 % |
| runtime 0.15 (vx / wz) | 0.194 / −0.190 | 0.185 / +0.165 | 0.196 / +0.446 | 0.177 / −0.133 |
| runtime 0.25 (vx / wz) | 0.285 / −0.148 | 0.276 / +0.080 | 0.265 / +0.622 | 0.266 / −0.041 |

Latency 20 / 60 ms: 94 / 73 %. Friction 0.5 / 1.5: wz −0.081 / −0.018. Knee p95 0.17–0.24 N·m.
Runtime: fell false, latched 0. Deadline misses were 4 at 0.15 (period_max 53.8 ms, while a VM used
several cores) and 0 at 0.25. MLP parity 1.3e-6.

## Verdict
| row | MAC-25 | pass? |
|---|---|---|
| \|wz\| ≤ 0.08 at 0.15 / 0.25, DR off | 0.047 / 0.006 | yes |
| \|wz\| ≤ 0.08 at 0.15 / 0.25, DR on | 0.052 / 0.028 | yes |
| \|wz\| ≤ 0.05 at zero command | 0.034 / 0.020 | yes |
| turn 0.8 → wz ≥ 0.5 | 0.579 / 0.523 | yes |
| 40 ms latency ≥ 80 % of vx | 83 % | yes |
| 0 falls | 0 | yes |

**Accepted. MAC-25 replaces MAC-14 as the biped_v1 reference policy.** The mirror loss
does what augmentation could not on a warm start: it pulls the policy toward symmetry
gradually, without breaking the PPO ratio.

Still open: through the runtime, the 0.15 case yaws −0.133 rad/s (MAC-14: −0.190). Why the
runtime path yaws more than the Python env at 0.15 (−0.047) is the remaining gap.
The 4 deadline misses should be rerun on a quiet machine before hardware.

Files: `mac25/` holds `run.json`, `eval.json`, `report.json` (history every 10th iteration,
now with `loss_*`), `runtime-v0.15.json`, `runtime-v0.25.json`, `mac25-mlp.json` (local
paths shortened).
policy.onnx sha256 `3789a53b243f32d81492c42d0e8c793c31c10a35ddf26a3f9d15a4872745a52b` (not committed).
