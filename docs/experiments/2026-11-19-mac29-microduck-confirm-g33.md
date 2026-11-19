# Microduck: does MAC-28's G3.3 pass hold? M5: no (MAC-29, WP3-17)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 3e276f0 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate
       --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 0.5 (MAC-27 flags), 256 envs,
       planned 1000 it from mac28-seed0/model_17300.pt, 8 workers, seed 0 (a VM used ~4–7 cores)
```

## Training
Falls per iteration (100-it means) were 2.15, 1.57, 1.28, 1.53, 1.47, 1.64. The 200-iteration mean
after the first 100 iterations reached 1.55 at iteration 600, so **the falls rule fired**, and the run
script stopped training at iteration 17900. loss_symmetry went 0.0148 → 0.0111.

## Checkpoints (BAM; G1.3 = 4 × 10 s; 0 falls in every G1.3 case)
| ckpt | 0.2 vx (G1.3) | wz at 0.2 | back 0.15 | sweep 0.3 (wz) | sweep 0.4 (wz) | turn 1.0 wz |
|---|---|---|---|---|---|---|
| MAC-28 17300 | 0.160–0.161 | −0.011 | −0.126 | 0.255 (−0.03) | 0.342 (+0.01) | 0.001 |
| 17600 | 0.132–0.133 | −0.050 | −0.154 | 0.179 (−0.12) | 0.198 (−0.18) | 0.002 |
| 17700 | 0.081–0.092 | −0.132 | −0.084 | 0.216 (−0.05) | 0.221 (+0.12) | 0.002 |
| 17800 | 0.128–0.135 | −0.097 | −0.119 | 0.168 (−0.15) | 0.220 (−0.12) | 0.002 |
| 17900 | 0.125–0.129 | +0.025 | −0.120 | 0.141 (+0.14) | 0.239 (+0.08) | 0.001 |

Best is 17600. `eval_velocity_env` at 0.4 gives 0.295 / 0.303, 0 falls. It stands 4 × 20 s and
4 × 30 s with no falls (tilt 1.6°). At 0.2 for 4 × 20 s it gives 0.113–0.116, 0 falls.

## Verdict
**Not confirmed.** No checkpoint reaches 0.14 m/s at 0.2 with |wz| < 0.05, and the training falls
rule fired. MAC-28's model_17300 was a snapshot: from it, the gait starts to drift again
(|wz| up to 0.18 at 0.4), and 0.2 settles at 63–66 %, the MAC-27 level. The drift changes
sign from checkpoint to checkpoint, so the policy is still moving around a lot. G3.3 on BAM
needs a more stable training setup (for example a lower learning rate or larger batch for the
fine-tune, more envs, or a higher mirror-loss weight), not just more iterations.

Files: `mac29/ckpt-<it>/` holds `g1-3.json` and `sweep-bam.json`, plus `eval.json` and
`long-g13.json` for 17600. `history_17300-17900.json` is the training history, and
`watch.log` is the falls watchdog output. ONNX sha256: 17600
`e6401b86666adc6eb77afcb260fe8a0ff935f9123f0f2d241583a34a53b347e4` (not committed).
