# biped_v1 DR ablation on the M5: action delay stops the gait (MAC-13, WP4-18)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0), 2026-11-19
revision: 75d5eea (code identical to main 6164ca5), git_dirty false
train: biped_v1, --dr, 256 envs, 3000 it, 8 workers, seed 0 (as MAC-11)
eval: eval_biped.py --dr --robustness --envs 8 --seconds 10
```

## Question
MAC-08 (old DR) walked; MAC-11 (WP4-16 DR) only stood. WP4-16 changed two things:
0–40 ms action delay, and foot friction that really reaches 0.7. Each run turns one off.

## Results
| | A: `--dr-delay-max 0` | B: `--dr-friction 1.0 1.3` |
|---|---|---|
| walks? | **yes** | no, stands |
| vx at 0.15 / 0.25 (DR on) | 0.101 / 0.163 (67 / 65 %) | 0.000 / 0.000 |
| vx at 0.15 / 0.25 (DR off) | 0.135 / 0.216 | 0.000 / 0.000 |
| falls (all cases) | 0 | 0 |
| friction_0.5 / 1.5 | 0.134 / 0.136 | 0.000 / 0.000 |
| latency 20 / 40 / 60 ms | 0.090 / 0.063 / 0.046 | 0.000 each |
| pushes 0.3 m/s fwd / side | 0 falls, 0.137 / 0.135 | 0 falls, 0.000 |
| knee p95 / max (DR on, 0.15) | 0.195 / 0.52 N·m | 0.066 / 0.26 N·m |
| air_time (250-it mean) | 0.005 → 0.025 (it 500) → 0.030 (it 2750) | 0.0067 → 0.0001 (it 500); ≈ 0 from it 629 |
| reward/step first → last 250 it | 0.022 → 0.127 | 0.018 → 0.098 (flat from ~1500) |
| wall time | 2383 s | 2190 s |

## Conclusion
- **The DR action delay (0–40 ms) causes the standing local optimum.** Lower friction
  alone (A) still lets the policy walk. With the delay on (B, and MAC-11), the feet
  stop leaving the ground by about iteration 600, and standing earns about 0.10 reward/step.
- A meets G4.3 first pass: ≥ 50 % of the command at 0.15 and 0.25 with DR on, no falls.
- A still loses speed with latency it never saw: 42 % at 40 ms (the MAC-11 target is
  ≥ 50 %). So the delay belongs in training, but it needs a way past the standing
  optimum. Options for the next cloud WP: a delay curriculum (0 at first, ramp to 40
  ms after the gait forms), fine-tuning A with the delay on, or a reward change so
  that standing at a nonzero command no longer pays ~0.10.
- Knee max hits 0.52 N·m (the clip) in A, but p95 stays around 0.2. No change to
  the WP4-05 servo note.

Files: `mac13/a/`, `mac13/b/` hold `run.json`, `eval.json` and `report.json` (the
`history` keeps every 10th iteration, `history_stride: 10`, to keep the repo small).
policy.onnx sha256: A `9a2823b3fe2e4377e706050ed805f092de42d0cdad2203d628e3ff5ad0b3a7f6`,
B `09b31691ae7027f916c24d443ca6a728b8a3b80ede7e51fbc35eb16e5b24a3b1` (not committed).
