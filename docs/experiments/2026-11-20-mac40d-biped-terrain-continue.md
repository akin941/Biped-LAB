# biped_v1 terrain policy, 800 more iterations at 10 mm, M5 (MAC-40d, WP4-32)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: code 696d466 (worktree HEAD 7f998c9), box-tile floor
train: from mac40c/model_6700.pt, MAC-40c flags (--rough-floor 0.010 --rough-floor-ramp-it 0 --mirror-loss-coeff 1.0
       --desired-kl 0.005, STS3215 curve, yaw std2 0.25, --dr), --iters 800 --save-every 50; falls rule 1.5/it
eval: eval_biped --terrain --robustness --dr --envs 16 --seconds 20, STS3215 curve, every saved checkpoint
```

All 800 iterations ran in 653 s. The falls rule did not fire (falls/it 0.70–1.29, 100-it means).

## Every checkpoint (16 envs × 20 s)
| ckpt | flat \|wz\| max | 40 ms | 10 mm falls | 20 mm falls | pass |
|---|---|---|---|---|---|
| 6750 | 0.085 | 86 % | 0 | 5 | |
| **6800** | **0.013** | 95 % | 0 | 7 | ✔ |
| **6850** | **0.072** | 92 % | 0 | 9 | ✔ |
| 6900 | 0.116 | 93 % | 0 | 4 | |
| 6950 | 0.240 | 103 % | 0 | 5 | |
| 7000 | 0.383 | 100 % | 0 | 10 | |
| 7050 | 0.396 | 105 % | 0 | 5 | |
| 7100 | 0.211 | 104 % | 0 | 6 | |
| 7150 | 0.225 | 98 % | 0 | 2 | |
| 7200 | 0.268 | 91 % | 0 | 3 | |
| 7250 | 0.464 | 95 % | 0 | 7 | |
| 7300 | 0.475 | 96 % | 0 | 6 | |
| 7350 | 0.336 | 94 % | 0 | 7 | |
| 7400 | 0.299 | 94 % | 0 | 4 | |
| 7450 | 0.293 | 104 % | 0 | 3 | |
| 7500 | 0.096 | 93 % | 0 | 4 | |

All 16 checkpoints have 0 flat falls and 0/16 falls at 5 mm. The worst |wz| is almost always the 0.25 command with DR off, and it is positive.

## Verdict
- **Not met.** 6800 and 6850 pass in a row, and no run reaches three. With MAC-40c's history, 6650 and 6700 pass, 6750 misses at 0.085, then
  6800 and 6850 pass.
- **Terrain is fully steady.** 10 mm 0/16 on every checkpoint, 20 mm 2–10/16, and 40 ms 86–105 %.
- **The flat yaw is what fails, and it gets worse with training.** It climbs to +0.3…+0.48 rad/s at the 0.25 command by 7000–7450.
  Training only on the 10 mm floor pulls the flat 0.25 gait into a turn, and mirror loss 1.0 doesn't stop it.
- **Next (cloud):** mix flat and rough episodes (e.g. 30–50 % flat), or add flat-floor yaw tracking during terrain training. Repeat with
  checkpoints every 50 iterations. No terrain policy is named; 6400 / 6800 stay single-checkpoint candidates.

ONNX sha256: 6800 `6e72d75c4c8335ae4c7e40f97ebc96a9d0cf079923f56d1e835cd87f6df5bf8a`,
6850 `abddf75d9a68c739eaa6decd7bc729b0364a41bb0c12983c280ffc9124afbb2a` (not committed).
Files: `mac40d/eval-<it>.json` (6750–7500), `mac40d/report.json` (history every 10th iteration), `mac40d/run.json`, `mac40d/watch.log`.
