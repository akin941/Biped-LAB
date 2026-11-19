# Microduck: speed-relative tracking kernel on BAM, M5: slow commands step, fast ones regress (MAC-26, WP3-15)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 2ac6883 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate
       --track-lin-ref-speed 0.4, 256 envs, 2000 it (13000 -> 15000), 8 workers, seed 0,
       --resume mac24-seed0/model_13000.pt; 3124 s (a VM used ~5–7 cores)
```

## Results (BAM; 10 s per case; vx m/s, dy m, wz rad/s; 0 falls everywhere)
| case | MAC-24 | **MAC-26** |
|---|---|---|
| fwd 0.1 | 0.0002 | 0.0003 |
| fwd 0.2 (G1.3, 4 repeats) | 0.0004 | **0.091–0.099** (48 %), dy −1.10, wz −0.16 |
| fwd 0.3 | 0.193, dy +0.57, wz +0.09 | 0.110, dy −1.80, wz −0.17 |
| fwd 0.4 (sweep) | 0.260, dy +1.46, wz +0.07 | 0.116, dy −2.24, wz −0.19 |
| fwd 0.4 (eval_velocity_env) | 0.316 | 0.319 |
| back 0.15 (G1.3) | ≈ 0 | **−0.073** (48 %) |
| back 0.3 | −0.211 | −0.210 |
| lateral 0.3 (dy) | +1.06 | +0.64 |
| turn 1.0 (wz) | 0.965 | **0.002** |
| turn 0.5 (G1.3) | ≈ 0 | ≈ 0 |

Training (250-it means): falls/it 1.07–1.45, and the 200-iteration mean never passed 1.5, so the run
was not stopped. `air_time` went 0.0169 → 0.0181. The tracking term stayed flat at about 0.017 on the
tighter kernel. Reward/step fell 0.030 → 0.024 because the tighter kernel shrinks the scale.

**G3.3 not met.** 0.2 reaches 48 % of the command, and 70 % is needed.

## Reading
- The speed-relative kernel does what it was meant to do at slow speeds. For the first time,
  0.2 and backward 0.15 both step with no standing.
- It cost the faster commands. 0.3 and 0.4 (sweep) dropped to about 0.11, and the gait now
  curves right at every forward speed (wz −0.16 to −0.19, dy up to −2.2 m). Turning at 1.0
  is gone. `eval_velocity_env` at 0.4 still reads 0.32, so in the sweep the curve, not the
  stride, limits forward progress.
- 2000 iterations from MAC-24 traded the high end for the low end. The yaw drift is now the
  main error at every forward speed. Next ideas (cloud): yaw pressure on Microduck too (the
  biped's MAC-25 mirror loss worked), or a longer run with this kernel if the tracking term
  starts rising.

Files: `mac26/` holds `run.json` (resume path shortened), `eval.json`, `g1-3.json`,
`sweep-bam.json` and `report.json` (history every 10th iteration).
policy.onnx sha256 `fd220aaa52439603bdf0bead51187b0dd5b2b6a42813b39a515ecac39a55d35a` (not committed).
