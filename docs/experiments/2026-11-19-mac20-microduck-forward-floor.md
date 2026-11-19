# Microduck on BAM with slow forward commands in training, M5: threshold moved up (MAC-20, WP3-13)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 2dc888c (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1, 256 envs,
       2000 it (8000 -> 10000), 8 workers, seed 0, --resume mac17-seed0/model_8000.pt; 2287 s
```

## Results (BAM)
| forward command | 0.1 | 0.2 | 0.3 | 0.4 | dy at 0.3 / 0.4 (10 s) |
|---|---|---|---|---|---|
| MAC-15 | 0.0001 | 0.0000 | 0.120 | 0.149 | – |
| MAC-17 | 0.0001 | 0.0002 | 0.078 | 0.131 | −0.996 / −1.123 m |
| **MAC-20** | 0.0002 | 0.0002 | 0.0002 | 0.170 | −0.001 / +0.739 m |

The 0.2 values come from G1.3 (4 × 10 s), the others from `sweep-bam.json`. Backward, lateral and turn
are all ≈ 0. No falls anywhere. `eval_velocity_env` at 0.4: vx 0.217 / 0.200 (noise on /
off), 0 falls. Reward/step rose 0.099 → 0.103. `air_time` fell 0.0145 → about 0.0125.

**G3.3 not met.** The walking threshold moved up: the policy now also stands at 0.3.

## Conclusion
- More slow forward commands in training did not teach a slow gait. They taught the
  policy to stand at more commands. Below the speed the BAM gait can reach, standing
  still pays more than a poor attempt at tracking, so each extra low-command sample
  strengthens standing.
- The lateral drift at 0.4 changed sign (−1.1 → +0.7 m), which shows the gait is still
  shifting between runs.
- Sampling and kernel knobs (MAC-17, MAC-20) don't break the standing optimum. Next
  ideas (cloud): stop paying for standing at a nonzero command (gate upright/pose/head
  on tracking, or add a no-progress penalty), a gait-phase or clock reference, or a
  speed curriculum that starts at 0.4 and lowers the command once stepping is stable.

Files: `mac20/` holds `run.json` (resume path shortened), `eval.json`, `g1-3.json`,
`sweep-bam.json` and `report.json` (history every 10th iteration).
policy.onnx sha256 `b8497e7a74d90e1b938fc82d779a40f78282b9b3715d8ec88671c7dbbe15eac3` (not committed).
