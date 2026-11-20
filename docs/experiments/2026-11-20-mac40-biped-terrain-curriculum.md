# biped_v1 terrain curriculum on box tiles, M5: the ramp is too steep (MAC-40, WP4-32)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: 696d466 (main, box-tile floor), git_dirty false
train: biped_v1 from mac37-b/model_5000.pt, --dr --track-yaw-std2 0.25 --symmetry loss
       --mirror-loss-coeff 0.5 --torque-limit 1.91 --motor-no-load-speed 5.45 --rough-floor 0.02
       --rough-floor-ramp-it 600 --desired-kl 0.005, 256 envs, 8 workers (planned 1500 it)
eval: eval_biped --terrain --robustness --dr --envs 16 --seconds 20, STS3215 curve
```

## Floor check (before training)
MAC-37 B at 0.15 m/s, 16 envs × 20 s: the plane, tiles at 10 µm and tiles at 2 mm all give 0/16 falls with
DR off and on, at the same speeds. The tile floor matches the plane, so the two earlier void runs' problems
(8 m field; height-field contacts) are gone.

## Training
| it (ramp height) | 0 (0.4 mm) | 25 | 50 | 75 | 100 (3.7) | 125 | 150 (5.4) | 175 (6.2) | 200 (7.1) | 225 (7.9) | 250 (8.7) | 275 (9.6) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falls / it | 1.28 | 0.28 | 0.24 | 0.16 | 0.20 | 0.44 | 0.56 | 1.24 | 2.00 | 3.48 | 3.64 | 4.04 |

Iterations 200–300 averaged 3.29 falls per iteration, so the run stopped at 5300 (the "> 3/it after it 300" rule).
The last good checkpoint is model_5200.

## Results (16 envs × 20 s; falls / vx / wz; vx and wz averaged over envs still standing)
| case | model_5200 | MAC-37 B (same tiles) |
|---|---|---|
| rough 5 mm | 0 / 0.198 / +0.048 | 0 / 0.165 / −0.009 |
| rough 10 mm | **13** / 0.158 / +0.019 | 15 / 0.105 / −0.080 |
| rough 20 mm | 16 / −0.038 / −0.310 | 16 / −0.065 / +0.216 |
| flat 0.15, DR off / on | 0 / 0.196 / +0.048 ; 0 / 0.170 / +0.016 | 0 / 0.163 / +0.016 ; 0 / 0.145 / +0.013 |
| flat 0.25, DR off / on | 0 / 0.272 / **+0.130** ; 0 / 0.247 / **+0.116** | 0 / 0.244 / −0.006 ; 0 / 0.219 / +0.006 |
| zero command wz, off / on | −0.103 / −0.069 | +0.009 / +0.025 |
| 40 ms latency (of 0.15) | 97 % | 81 % |

All flat robustness cases (friction, pushes, latency) have 0 falls for both policies.

## Verdict
**WP4-32 not met.** `rough_10mm` is 13/16 falls (0 needed), and the fine-tune breaks the flat 0.25 yaw row
(|wz| 0.13 / 0.12). Falls stay low up to about 5 mm and then climb with the ramp. A ramp of 20 mm over 600 it
(about 3.3 mm per 100 it) is steeper than the policy can adapt to. The 200 iterations bought a small
gain at 10 mm (15 → 13 falls) and a faster, drifting flat gait. Next: MAC-40b, a slower, lower ramp
(12 mm over 1500 of 2500 it) from the same checkpoint.

Files: `mac40/eval.json` (model_5200), `mac40/eval-mac37b-tiles.json`, `mac40/tile-check.json`,
`mac40/history_5000-5300.json`, `mac40/watch.log` (paths shortened).
