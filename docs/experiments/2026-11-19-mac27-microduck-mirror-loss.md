# Microduck: mirror loss on BAM, M5: the curve is gone, 0.2 reaches 63 % (MAC-27, WP3-16)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 6c04cbe (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate
       --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5, 256 envs,
       2000 it (15000 -> 17000), 8 workers, seed 0, --resume mac26-seed0/model_15000.pt;
       2746 s (a VM used ~6–7 cores)
```

## Results (BAM; 10 s per case; vx m/s, dy m, wz rad/s; 0 falls everywhere)
| case | MAC-24 | MAC-26 | **MAC-27** |
|---|---|---|---|
| fwd 0.1 | 0.0002 | 0.0003 | 0.0005 |
| fwd 0.2 (G1.3, 4 repeats) | 0.0004 | 0.091–0.099, wz −0.16, dy −1.10 | **0.124–0.127** (63 %), wz −0.002, dy +0.17 |
| fwd 0.3 | 0.193, wz +0.09 | 0.110, wz −0.17, dy −1.80 | **0.200**, wz −0.020, dy −0.01 |
| fwd 0.4 (sweep) | 0.260, wz +0.07 | 0.116, wz −0.19, dy −2.24 | **0.274**, wz −0.018, dy −0.18 |
| fwd 0.4 (eval_velocity_env) | 0.316 | 0.319 | 0.292 |
| back 0.15 (G1.3) | ≈ 0 | −0.073 | **−0.123** (82 %) |
| back 0.3 | −0.211 | −0.210 | −0.182 (dy +0.79, wz −0.14) |
| lateral 0.3 (dy) | +1.06 | +0.64 | +0.49 |
| turn 1.0 / turn 0.5 (wz) | 0.965 / ≈ 0 | 0.002 / ≈ 0 | 0.003 / ≈ 0 |

Training (250-it means): loss_symmetry 0.0203 → 0.0117. Falls/it ranged 1.05–1.45, and the
200-iteration mean never passed 1.5. `air_time` stayed at about 0.018 and tracking at about 0.017.

**G3.3 not met, but close.** At 0.2 the policy reaches 63 % of the command, and ≥ 70 % is needed.

## Reading
- The mirror loss removed the curving gait. |wz| at 0.2 / 0.3 / 0.4 fell from 0.16–0.19
  to 0.002–0.020, and the sideways drift almost vanished. Without the curve, 0.3 and 0.4
  are back above MAC-24, and 0.2 rose from 48 % to 63 %.
- What's left: 0.2 needs about 10 % more speed, and turning is still missing (0.5 and 1.0
  give ≈ 0, lost since MAC-26). Backward 0.3 still yaws (−0.14).
- Next: continue this run (the symmetry loss is still falling), or add a turn command
  weight. The turning loss came in with the speed-relative kernel (MAC-26), so the yaw
  kernel may need the same speed scaling.

Files: `mac27/` holds `run.json` (resume path shortened), `eval.json`, `g1-3.json`,
`sweep-bam.json` and `report.json` (history every 10th iteration).
policy.onnx sha256 `f62cbb0b26a603ae3e5f2f2a512d23761a628b75cdc1d8b55f5aaa88c6dd937f` (not committed).
