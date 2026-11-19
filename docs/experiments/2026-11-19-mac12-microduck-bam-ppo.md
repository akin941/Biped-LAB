# Microduck PPO on the BAM actuator, M5: stands, does not walk (MAC-12, WP3-10, G3.3)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 75d5eea (code identical to main 6164ca5), git_dirty false
train: microduck, --actuator bam --dr, 256 envs, 3000 it, 8 workers, seed 0, init_std 1.0
```

## Results
- Timing (20 it, BAM): 4 workers 3392 samples/s, 8 workers 3546 → 8. The machine was
  loaded (load avg 12–18) because MAC-13 was running earlier in the same pipeline.
- Training: 3000 it, 18.4 M samples, 3496 s (58 min). Reward/step rose 0.0427 → 0.0839.
- `air_time` (250-it means): 0.0071 → 0.0016 (it 250) → 0.0009 (it 500) → 0.0004 (it 2750).
  It never reaches exactly 0, but the feet hardly leave the ground after about iteration 300.
- G1.3 with BAM (`bipedlab eval-policy`, 4 repeats × 10 s): no falls in any scenario,
  stand_still tilt < 1°, but `walk_forward_0.2` vx −0.0001 m/s (0 %), backward and turn
  likewise ≈ 0.
- `eval_velocity_env.py --actuators bam` (0.4 command): vx ≈ −0.00005 m/s, 0 falls,
  `track_linear_velocity` 0.008/step versus `upright` 0.040 + `head_pose_tracking` 0.036.

**G3.3 is not met.** The policy found a stable standing posture and earns its reward
from the upright, pose and head-pose terms.

## Notes
- This is not the biped_v1 DR action-delay effect (MAC-13). The Microduck env has
  no DR action delay. The BAM actuator plus this reward mix reaches the same standing
  optimum. Upstream trains with BAM too (ADR-0011), so likely causes are the reward
  weights, the BAM drop gain, or too few iterations (the reward was still rising).
- Candidates for the next cloud WP: reward ablation against upstream weights, warm-start
  from the MAC-06 MJCF policy (it walks) and fine-tune on BAM, or a gait-phase/air-time
  shaping term.

Files: `mac12/` holds `run.json`, `eval.json`, `g1-3.json`, both timing JSONs and
`report.json` (the `history` keeps every 10th iteration, `history_stride: 10`).
policy.onnx sha256 `1216c60502330ed0ac16fec745ff4036218447843d603c38908f51719f905fc7` (not committed).
