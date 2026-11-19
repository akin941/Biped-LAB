# Microduck posture-gate run continued on BAM, M5: better at large commands, stands at small ones (MAC-24, WP3-14)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 61caa14 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate,
       256 envs, 3000 it (10000 -> 13000), 8 workers, seed 0, --resume mac22-seed0/model_10000.pt;
       4292 s (a VM used ~4–6 cores, load avg 10–33)
```

## Results (BAM; 10 s per case; vx m/s, dy m, wz rad/s; 0 falls everywhere)
| case | MAC-22 | **MAC-24** |
|---|---|---|
| fwd 0.1 | 0.0005 | 0.0002 |
| fwd 0.2 (G1.3, 4 repeats) | 0.061, dy −1.16, wz −0.19 | **0.0004**, dy −0.00, wz −0.00 |
| fwd 0.3 | 0.219, dy +0.55, wz +0.02 | 0.193, dy +0.57, wz +0.09 |
| fwd 0.4 (sweep) | 0.120, dy +2.18, wz +0.19 | **0.260**, dy +1.46, wz +0.07 |
| fwd 0.4 (eval_velocity_env) | 0.311 | 0.316 |
| back 0.3 | −0.150 | **−0.211** |
| lateral 0.3 (dy) | +0.008 | **+1.06** |
| turn 1.0 (wz) | 0.757 | **0.965** |
| back 0.15, turn 0.5 (G1.3) | ≈ 0 | ≈ 0 |

Training (500-it means): reward 0.0375 → 0.0413, tracking 0.0236 → 0.0247, air_time ≈ 0.015,
falls/it 1.12 → 0.98 (one bucket at 1.25), so the run was not stopped.

**G3.3 not met.** At 0.2 the policy is back to standing (MAC-22 stepped at 0.061).

## Reading
- Above about 0.3 the gait got better everywhere: faster at 0.4, less sideways drift,
  backward faster, and lateral stepping and turning work. Below that, every small command
  (fwd 0.1–0.2, back 0.15, turn 0.5) is still answered by standing.
- MAC-24's question was whether vx at 0.2 and the drift moved. The drift shrank, but vx at 0.2
  moved down. So more training at the same sampling won't reach G3.3. The small
  commands need their own pressure: for example a gate that is stricter at small
  commands, a minimum-motion term while a command is nonzero, or a curriculum that
  lowers the command from 0.3.

## Correction to earlier records
While looking up the yaw rates I found that MAC-15 and MAC-22 recorded the `turn_1.0`
sweep case as "≈ 0" because the vx column was read. The yaw rates are 0.941 (MAC-15)
and 0.757 rad/s (MAC-22). This PR fixes the MAC-15 and MAC-22 handoff blocks and
notes. MAC-17 (−0.001) and MAC-20 (0.001) were right. Lateral 0.3 was ≈ 0 in all
runs before MAC-24.

Files: `mac24/` holds `run.json` (resume path shortened), `eval.json`, `g1-3.json`,
`sweep-bam.json` and `report.json` (history every 10th iteration).
policy.onnx sha256 `5c5e9f3566f78a96d73d4f03f611d0ef9fee06af2dcc8853ae3cd6f8c2e64bb7` (not committed).
