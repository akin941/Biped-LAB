# biped_v1 terrain policy held at 10 mm with mirror loss 1.0, M5 (MAC-40c, WP4-32)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: code = 696d466 (worktree HEAD 7f998c9 = 696d466 + MAC-40 docs), box-tile floor
train: from mac40b/model_6400.pt; --rough-floor 0.010 --rough-floor-ramp-it 0 --mirror-loss-coeff 1.0
       --iters 1000 --save-every 50; otherwise MAC-40b (--desired-kl 0.005, STS3215 curve, yaw std2 0.25, --dr)
eval: eval_biped --terrain --robustness --dr --envs 16 --seconds 20, STS3215 curve, on every saved checkpoint
```

## Training
Falls per iteration (100-it means) were 0.92, 0.94 and 1.08. The falls rule (> 1.0 over 100 it, on from the start) fired at
iteration 300, after 4 minutes. The run stopped at model_6700, and the last good checkpoint is model_6650.

## Checkpoints (16 envs × 20 s)
| ckpt | flat \|wz\| max | 40 ms | flat falls | rough 5 / 10 / 20 mm falls | vx at 10 mm | pass |
|---|---|---|---|---|---|---|
| 6450 | 0.171 | 78 % | 0 | 0 / 0 / 8 | 0.140 | |
| 6500 | 0.215 | 84 % | 0 | 0 / 0 / 10 | 0.133 | |
| 6550 | 0.272 | 93 % | 0 | 0 / 0 / 7 | 0.159 | |
| 6600 | 0.148 | 90 % | 0 | 0 / 0 / 3 | 0.152 | |
| **6650** | **0.045** | 91 % | 0 | 0 / 0 / 9 | 0.145 | ✔ (last good) |
| **6700** | **0.042** | 93 % | 0 | 0 / 0 / 7 | 0.158 | ✔ (at the rule) |

## Verdict
- **Not met:** two consecutive passing checkpoints (6650, 6700), and three are needed. The run stopped at 300 of 1000 iterations.
- **Terrain is steady.** 10 mm gives 0/16 falls on every checkpoint, so holding the floor at 10 mm keeps the skill.
- **The flat yaw is what swings.** It sits at −0.17 to −0.27 rad/s at 6450–6550, then settles to about 0.04 by 6650–6700, which
  suggests the mirror loss 1.0 is taking hold.
- **The falls rule is about the training floor's own fall rate.** At a held 10 mm the policy falls about 0.9–1.1 times per
  iteration in training, while the 10 mm eval floor gives 0/16. Suggestion: rerun from model_6650 with the rule at
  1.5 / it (or relative to the 10 mm training baseline), so the run can show three or more passing checkpoints in a row.

ONNX sha256: model_6650 `3254104cf1fe4b671b686f32bcdd45b5785740faaca5933c88ba58bd4bc1d15f`,
model_6700 `0881422d097af356194116fcd511f972269b5944adae5c79667e7ca851aae50e` (not committed).
Files: `mac40c/eval-<it>.json` (6450–6700), `mac40c/history_6400-6700.json`, `mac40c/watch.log`.
