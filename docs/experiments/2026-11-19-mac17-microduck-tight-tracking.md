# Microduck on BAM with a tighter linear tracking kernel, M5: threshold unchanged (MAC-17, WP3-12)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 441e731 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05, 256 envs, 2000 it (6000 -> 8000),
       8 workers, seed 0, --resume mac15-seed0/model_6000.pt; 2308 s
```

## Results
| forward command (BAM) | 0.1 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|
| MAC-15 vx | 0.0001 | 0.0000 | 0.120 | 0.149 |
| **MAC-17 vx** | 0.0001 | 0.0002 | 0.078 | 0.131 |

The 0.2 values come from G1.3 (`g1-3.json`, 4 × 10 s), the others from `sweep-bam.json`.
Backward 0.3, lateral 0.3 and turn 1.0 are all ≈ 0. No falls anywhere.
`eval_velocity_env` at 0.4: vx 0.199 / 0.193 (noise on / off), 0 falls.
Reward/step rose 0.091 → 0.097. `track_linear_velocity` went 0.0198 → ~0.021. It is measured on the
tighter kernel, so it is not comparable with MAC-15's 0.025. `air_time` was steady at 0.013–0.014.

**G3.3 not met.** The command threshold for walking stays between 0.2 and 0.3 m/s,
and the 0.3 point got slower.

## Conclusion
- Tightening the linear kernel alone isn't enough. At 0.2, standing still earns 45 % of
  the tracking term plus the full upright, pose and head terms, so the push towards
  stepping stays weak, and 2000 iterations from the MAC-15 policy didn't find it.
- Next ideas (cloud): oversample low forward commands (0.1–0.25) and backward/turn
  commands during the BAM fine-tune, add a penalty for standing still at a nonzero
  command (or gate the upright/pose rewards on tracking), or a gait-phase reference.

Files: `mac17/` holds `run.json` (resume path shortened), `eval.json`, `g1-3.json`,
`sweep-bam.json` and `report.json` (history every 10th iteration).
policy.onnx sha256 `e39b7180d6cf072ff6fb19c4d452308e22c935e0fdfea5617e7a049237a7548e` (not committed).
