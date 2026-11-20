# biped_v1 terrain training with 40 % flat episodes, M5 (MAC-41, WP4-33)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: 7a1fa16 (main), git_dirty false; pre-flight pytest -k flat_fraction passed
train: from mac40d/model_6800.pt; MAC-40d flags + --rough-floor-flat-fraction 0.4, --iters 800 --save-every 50;
       falls rule 1.5/it after 100 it
eval: eval_biped --terrain --robustness --dr --envs 16 --seconds 20, STS3215 curve, every saved checkpoint
```

All 800 iterations ran in 799 s. Training falls were 0.63–0.84 per iteration (MAC-40d: 0.70–1.29), so the rule did not fire.

## Every checkpoint (16 envs × 20 s)
| ckpt | flat \|wz\| max | 40 ms | flat falls | 10 mm falls | pass |
|---|---|---|---|---|---|
| 6850 | 0.113 | 86 % | 0 | 0 | |
| 6900 | 0.108 | 103 % | 0 | 0 | |
| 6950 | 0.100 | 92 % | 0 | 0 | |
| **7000** | **0.078** | 95 % | 0 | 0 | ✔ |
| 7050 | 0.143 | 97 % | 0 | 0 | |
| 7100 | 0.189 | 97 % | 0 | 0 | |
| 7150 | 0.084 | 95 % | 0 | 0 | |
| 7200 | 0.268 | 98 % | 1 | 0 | |
| 7250 | 0.466 | 88 % | 0 | 0 | |
| 7300 | 0.201 | 98 % | 0 | 0 | |
| 7350 | 0.092 | 89 % | 0 | 0 | |
| **7400** | **0.078** | 94 % | 0 | 0 | ✔ |
| 7450 | 0.169 | 94 % | 0 | 0 | |
| 7500 | 0.110 | 98 % | 0 | 0 | |
| 7550 | 0.202 | 111 % | 1 | 0 | |
| 7600 | 0.527 | 93 % | 0 | 0 | |

5 mm: 0/16 falls on every checkpoint. 20 mm: 4–12/16. The two flat falls were at 0.25 with DR on (7200, 7550).

## Verdict
- **Not met.** Only 7000 and 7400 pass (both at the limit, 0.078), and no two are consecutive. No terrain policy is named.
- The flat share helped in part: training falls are lower, and the flat yaw no longer drifts one way. It now swings to both signs
  (−0.20 … +0.53 rad/s, worst at 0.25). But the swing from one checkpoint to the next is as large as the 0.08 bar. Four more
  checkpoints sit just above it (0.084–0.110).
- Terrain stays steady: 10 mm 0/16 on all 16 checkpoints.
- Ideas (cloud): stronger flat yaw tracking at 0.25 (e.g. yaw std² 0.15), mirror loss 2.0, or averaging the weights of neighbouring
  checkpoints to damp the swing.

ONNX sha256: 7000 `ec2fa1ee406ddbcf67293a2fdce9aa22b4a2f1a3611b78358393527dee528bc0`,
7400 `7dbbb54112872eebdd9290e7dca3a2dbdac7efdb297c5078de90a44863a0a43d` (not committed).
Files: `mac41/eval-<it>.json` (6850–7600), `mac41/report.json` (history every 10th iteration), `mac41/run.json`, `mac41/watch.log`.
