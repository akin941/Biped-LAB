# Microduck on BAM with the posture gate, M5: it steps at 0.2–0.3 for the first time (MAC-22, WP3-14)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 8c701c2 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate,
       256 envs, 2000 it (8000 -> 10000), 8 workers, seed 0, --resume mac17-seed0/model_8000.pt; 1890 s
```

## Results (BAM; vx m/s, dy = displacement_y in 10 s)
| command | MAC-17 | MAC-20 | **MAC-22** |
|---|---|---|---|
| fwd 0.1 | 0.0001 | 0.0002 | 0.0005 |
| fwd 0.2 (G1.3) | 0.0002 | 0.0002 | **0.061–0.070**, dy −1.16 m |
| fwd 0.3 | 0.078, dy −1.00 | 0.0002 | **0.219**, dy +0.55 |
| fwd 0.4 (sweep) | 0.131, dy −1.12 | 0.170, dy +0.74 | 0.120, dy +2.18 |
| fwd 0.4 (eval_velocity_env) | 0.199 | 0.217 | **0.311** |
| back 0.3 | ≈ 0 | ≈ 0 | **−0.150** |
| lateral 0.3, back 0.15, turn 0.5 | ≈ 0 | ≈ 0 | ≈ 0 |
| turn 1.0 (wz) | −0.001 | 0.001 | 0.757* |

No falls in any eval. Training falls per iteration went 1.97 → 1.18 (MAC-17: about 1.3–1.5), so
the run was not stopped. `air_time` 0.0149 → 0.0159. Reward/step 0.041 → 0.038. The reward
"did not rise" only because the gated terms shrink the reward scale. The tracking term
rose 0.0222 → 0.0232.

**G3.3 not met** (0.2 gives 33 % of the command, and ±30 % is needed). **But the standing
optimum is broken.** This is the first BAM policy that steps at 0.2 and 0.3 and walks backward.

## Notes
- The gait curves: the walking yaw is −0.19 rad/s at 0.2, and dy reaches +2.2 m at 0.4. In the
  sweep, 0.4 (0.120) is slower than 0.3, while `eval_velocity_env` at 0.4 reads 0.31.
  The two harnesses differ in init and reset, so the sweep's 0.4 point is likely
  limited by the curve.
- Next: continue this run (it is still improving: tracking and air time rising, falls
  falling), and add yaw or lateral tracking pressure for the drift.

Files: `mac22/` holds `run.json` (resume path shortened), `eval.json`, `g1-3.json`,
`sweep-bam.json` and `report.json` (history every 10th iteration).
policy.onnx sha256 `4b2a2cc1bcf1b2511c1efa78c35f4606a43e90e352e64c4649bd7a8ac317259b` (not committed).

\* Correction (MAC-24): turn 1.0 was first recorded as ≈ 0 because the vx column was read; its yaw rate is 0.757 rad/s.
