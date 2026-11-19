# Microduck: speed-relative yaw kernel on the G3.3 policy, M5: turn 0.5 still ~0 (MAC-33, WP3-20)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 54151b4 (main), git_dirty false
train: microduck, MAC-32 flags (--mirror-loss-coeff 1.0 --desired-kl 0.005 --turn-fraction 0.4 ...)
       + --track-ang-ref-speed 1.0, 256 envs, 1500 it (19700 -> 21200), 8 workers, seed 0,
       --resume mac32-seed0/model_19700.pt; 1311 s
eval: G1.3 (BAM, 4 × 10 s) and eval_velocity_env --track-ang-ref-speed 1.0 --twist 0 0 1.0 /
      0 0 0.5 / 0 0 0 (noise off) on all 15 checkpoints 19800–21200
```

## Training
All 1500 iterations ran, and **the falls rule did not fire** (the 200-iteration mean was 0.94 at most).
loss_symmetry stayed flat (0.0074 → 0.0078). The yaw term **fell** (0.0107 → 0.0094), tracking stayed at
about 0.023, and reward went 0.031 → 0.028.

## Checkpoints (0 falls in every G1.3 case)
| ckpt | 0.2 vx | wz at 0.2 | G1.3 | turn 1.0 | turn 0.5 | zero |
|---|---|---|---|---|---|---|
| 19800 | 0.132–0.137 | −0.124 | | 0.691 | 0.008 | −0.001 |
| 19900 | 0.135–0.147 | −0.097 | | **1.041** | 0.008 | −0.000 |
| 20000 | 0.181–0.183 | +0.025 | ✔ | 0.694 | 0.004 | −0.000 |
| 20100 | 0.141–0.144 | −0.074 | | 0.397 | 0.002 | −0.001 |
| 20200 | 0.162–0.165 | −0.013 | ✔ | 0.335 | 0.003 | −0.001 |
| 20300 | 0.155–0.156 | +0.002 | ✔ | 0.573 | 0.004 | −0.000 |
| 20400 | 0.171–0.172 | −0.004 | ✔ | 0.364 | 0.006 | −0.000 |
| 20500 | 0.148–0.150 | −0.013 | ✔ | 0.472 | 0.003 | −0.000 |
| 20600 | 0.137–0.140 | +0.110 | | 0.254 | 0.002 | −0.000 |
| 20700 | 0.158–0.160 | −0.005 | ✔ | 0.398 | 0.002 | +0.000 |
| 20800 | 0.160–0.161 | +0.010 | ✔ | 0.399 | 0.009 | +0.000 |
| 20900 | 0.083–0.088 | +0.214 | | 0.355 | 0.006 | −0.000 |
| 21000 | 0.169–0.171 | −0.038 | ✔ | 0.337 | 0.005 | −0.000 |
| 21100 | 0.151–0.153 | +0.043 | ✔ | 0.400 | 0.003 | −0.000 |
| 21200 | 0.155–0.156 | +0.069 | | 0.604 | 0.011 | +0.000 |

## Best checkpoint: 20000
At 0.2 it gives 0.181–0.183, and 0.176–0.179 over 4 × 20 s. It stands 4 × 20 s and 4 × 30 s with no falls
(tilt ≤ 1.0°). In the sweep it gives 0.083 at 0.1 (its first steps at 0.1), 0.265 at 0.3 and 0.296 at 0.4,
−0.217 backward and dy +0.76 m lateral. **Turn 1.0 in the sweep fell at 3.96 s**, while the env eval
turned 0.694 without falls. Turning in place is not robust to the start state. The sweep starts from the
upstream init, while `eval_velocity_env` uses the env reset.

## Verdict
- **WP3-20 not met.** Turn 0.5 gives ≤ 0.011 on all 15 checkpoints, and ≥ 0.35 is needed. Turn 1.0 ≥ 0.7
  appears once (19900), on a checkpoint that fails G1.3. The speed-relative yaw kernel did not move
  turn 0.5, and the yaw term even fell. With the kernel scaling alone, the 0.5 command still isn't
  worth turning for.
- **G3.3 still holds** (no regression): 10 of 15 checkpoints meet G1.3, with runs 20200–20500 (four in a
  row), 20700–20800 and 21000–21100, and no falls-rule stop. 20000 is faster at 0.2 than
  model_19700, and it falls in the sweep's turn case.
- model_19700 stays the reference.

Files: `mac33/ckpt-<it>/` holds `g1-3.json` and `turn-eval.json` for all 15 checkpoints, plus `sweep-bam.json`
and `long-g13.json` for 20000. Also `report.json` (history every 10th iteration), `run.json` (resume path
shortened) and `watch.log`. 20000 ONNX sha256 `e87b7134d0a8d823f65eff3b92e5e23be45830012097e060a46195ecde6edbdb`
(not committed).
