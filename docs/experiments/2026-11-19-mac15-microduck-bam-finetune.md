# Microduck: MAC-06 gait fine-tuned on BAM, M5: walks only above 0.3 m/s (MAC-15, WP3-11, G3.3)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 59d9553 (main), git_dirty false
train: microduck, --actuator bam --dr, 256 envs, 3000 it (3000 -> 6000), 8 workers, seed 0,
       --resume mac06-seed0/model_3000.pt (MJCF-trained); 3502 s
load: a VM used ~4.4 cores during training (load avg 9-32)
```

## Results
- Training kept the gait. `track_linear_velocity` stayed at 0.0246 → 0.0256 per step
  (MAC-12 from scratch: ~0.008), `air_time` stayed at 0.013–0.015, and reward/step rose 0.087 → 0.099.
- `eval_velocity_env.py --actuators bam` at the 0.4 command: vx 0.203 / 0.197 m/s (noise
  on / off), 0 falls. The zero-action baseline falls 8/8.
- G1.3 (`eval-policy`, BAM, 4 × 10 s): no falls, tilt < 2°, but **vx 0.0000 at the 0.2
  command**. Backward 0.15 and turn 0.5 are also ≈ 0. **G3.3 not met.**

Diagnostic runs (1 repeat each, `mac15/diag/`):

| harness setting | command → vx (m/s) |
|---|---|
| BAM sweep | fwd 0.1 → 0.0001, fwd 0.3 → 0.120, fwd 0.4 → 0.149; back 0.3, lateral 0.3 → ≈ 0; turn 1.0 → wz 0.941* |
| BAM walking, `--init settle` | fwd 0.2 → −0.0002 |
| BAM walking, `--init straight` | fwd 0.2 → 0.011 |
| MJCF walking, `--init upstream` | fwd 0.2 → 0.117 |

## Conclusion
- Fine-tuning works better than training from scratch (MAC-12): the forward gait
  survives on BAM. But it only starts above a command threshold between 0.2 and 0.3 m/s,
  and only forwards (plus turning at 1.0 rad/s, see the correction below). The same weights on MJCF walk at 0.2, so BAM's weaker,
  back-EMF-damped response raises the threshold. G1.3 tests exactly 0.2, so it reads 0.
- Not a harness artefact: the init mode doesn't change the 0.2 result, and the sweep
  shows a clean threshold.
- Ideas for the next cloud WP: oversample low forward commands (0.1–0.25) and
  backward/turn commands in the BAM fine-tune, or lower the standing reward at nonzero commands
  (the same standing optimum as biped_v1 MAC-11/13B).

Files: `mac15/` holds `run.json` (resume path shortened to `<main checkout>/…`),
`eval.json`, `g1-3.json`, `report.json` (history every 10th iteration,
`history_stride: 10`) and `diag/*.json`.
policy.onnx sha256 `ddda1a28ae7a344b19d67ca49be86e251e33f37dbd7d23c05548e60b6a18c518` (not committed).

\* Correction (MAC-24): turn 1.0 was first recorded as ≈ 0 because the vx column was read. Its yaw rate
(`mean_gyro_z_second_half`) is 0.941 rad/s, so the policy does turn at the large command; G1.3's turn 0.5
stays ≈ 0.
