# Microduck: speed-relative yaw kernel on BAM, M5: stopped at it 300, that checkpoint passes G3.3 (MAC-28, WP3-17)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: d357331 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate
       --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5 --track-ang-ref-speed 1.0,
       256 envs, planned 2000 it from mac27-seed0/model_17000.pt, 8 workers, seed 0
       (a VM used ~6–7 cores)
```

## What happened
Falls per iteration (50-iteration means) went 1.72, 1.88, 2.30, 2.06, 1.86, 1.50. Over iterations
100–300 the mean was 1.93, above the handoff's stop rule (> 1.5 for 200 iterations after the
first 100). MAC-27 ended at 1.48. I stopped the training by hand at iteration 17300, after about
9 minutes. The yaw term stayed flat at about 0.007 per step.

I exported both saved checkpoints, model_17100 and model_17300, with the same flags
(`train_ppo.py --resume <ckpt> --iters 0`: the loop is skipped and `export_onnx` runs). Both
got the full MAC-27 eval set. Because G3.3 asks for "stands ≥ 20 s" and G1.3's stand case is
10 s, model_17300 also got 4 × 20 s and 4 × 30 s stand runs and 4 × 20 s at 0.2. These use
the same `microduck_policy.run()` as `bipedlab eval-policy`, with BAM and the same perturbed
starts.

## Results (BAM; vx m/s, wz rad/s; 0 falls in every case)
| case | MAC-27 (it 17000) | model_17100 | **model_17300** |
|---|---|---|---|
| fwd 0.2 (G1.3, 4 × 10 s) | 0.124–0.127, wz −0.002 | 0.067–0.068, wz −0.25 | **0.160–0.161 (80 %)**, wz −0.011 |
| fwd 0.2, 4 × 20 s | – | – | 0.159–0.161, 0 falls |
| stand 4 × 20 s / 4 × 30 s | – | – | 0 falls, tilt ≤ 1.6° |
| fwd 0.3 | 0.200, wz −0.020 | 0.145, wz −0.23 | **0.255**, wz −0.031 |
| fwd 0.4 sweep / eval_velocity_env | 0.274 / 0.292 | 0.117 / 0.358 | **0.342** / 0.355 |
| back 0.15 (G1.3) | −0.123 | ≈ 0 to −0.13 | −0.124 to −0.128 |
| back 0.3 | −0.182 | −0.215 | −0.235 |
| lateral 0.3 (dy) | +0.49 | +0.68 | +0.65 |
| turn 0.5 / 1.0 (wz) | ≈ 0 / 0.003 | 0.005 / 0.002 | **0.002 / 0.001** |

## Verdict
- **model_17300 meets G3.3 on the M5.** It stands 20 and 30 s without falling, and tracks
  0.2 m/s at 80 % (inside ±30 %) over 10 s and 20 s without falling. It is also the fastest
  and straightest checkpoint so far at 0.3 and 0.4.
- Caveats: it comes from a run the falls rule stopped (training falls 1.5–2.3 per
  iteration during these 300 iterations), and model_17100 shows that the policy was still
  moving a lot. The curve came back at it 17100 and was gone again by 17300. A longer or
  repeated run should confirm that this is stable, not a lucky snapshot.
- **Turning is still lost** (0.5 and 1.0 give ≈ 0). The speed-relative yaw kernel did not bring
  it back within 300 iterations, and the yaw term stayed flat.

Files: `mac28/ckpt-17100/` and `mac28/ckpt-17300/` hold `eval.json`, `g1-3.json` and
`sweep-bam.json`. `ckpt-17300/long-g13.json` holds the 20 / 30 s runs.
`history_17000-17300.json` is the training history (every iteration).
policy.onnx sha256: model_17300 `2a55c12e82be6f16e4b1b9bc09f41d74a919fc632b9ad8dc17b8de5489b26904`,
model_17100 `4b9ae2ec42116b3d941bbaea22d99f267a700021274e7f01e777d2d7247029ef` (not committed).
