# Microduck: stronger mirror loss on BAM, M5: G3.3 confirmed (MAC-32, WP3-19)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-19
revision: 8e60bf7 (main), git_dirty false
train: microduck, --actuator bam --dr --track-lin-std2 0.05 --forward-min-vx 0.1 --posture-gate
       --track-lin-ref-speed 0.4 --symmetry loss --mirror-loss-coeff 1.0 --desired-kl 0.005
       --turn-fraction 0.4, 256 envs, 1500 it (18500 -> 20000), 8 workers, seed 0,
       --resume mac31-seed0/model_18500.pt; 1316 s (a VM used ~4–6 cores)
```

## Training
All 1500 iterations ran, and **the falls rule did not fire** (the 200-iteration mean was 0.84 at most).
Falls per iteration were 0.57–1.05. loss_symmetry went 0.0109 → 0.0070. Reward/step rose 0.037 → 0.042,
the yaw term 0.0136 → 0.0154, and tracking stayed at about 0.023.

## G1.3 (BAM, 4 × 10 s) on every checkpoint, 0 falls everywhere
| ckpt | 0.2 vx | wz at 0.2 | pass | | ckpt | 0.2 vx | wz at 0.2 | pass |
|---|---|---|---|---|---|---|---|---|
| 18800 | 0.159 | −0.028 | ✔ | | 19500 | 0.157–0.159 | +0.050 | |
| 18900 | 0.150–0.152 | −0.015 | ✔ | | 19600 | 0.139–0.142 | +0.011 | |
| 19000 | 0.158–0.159 | +0.025 | ✔ | | 19700 | 0.168–0.170 | −0.023 | ✔ |
| 19100 | 0.134–0.142 | −0.097 | | | 19800 | 0.157–0.165 | +0.040 | ✔ |
| 19200 | 0.158–0.160 | +0.035 | ✔ | | 19900 | 0.165–0.166 | −0.034 | ✔ |
| 19300 | 0.141–0.142 | −0.009 | ✔ | | 20000 | 0.143–0.147 | +0.107 | |
| 19400 | 0.136–0.144 | +0.046 | | | | | | |

8 of 13 pass, in three runs: **18800–18900–19000, 19200–19300, 19700–19800–19900**. Of the
checkpoints that fail, 19500 is right at the |wz| limit (0.050), 19600 is just under the speed
bar (0.139), and the others fail on yaw.

## Passing checkpoints and the last one: sweep and turn
At 0.3 / 0.4 the speeds are 0.19–0.27 / 0.23–0.31. Turn 1.0 (env eval) reaches 0.03–0.47, with the best at
19200 (0.465) and 19700 (0.360). Turn 0.5 stays at about 0.01. The last checkpoint, 20000, turns 0.40 but
fails G1.3 on yaw (+0.107).

## Best checkpoint: 19700
At 0.2 it gives 0.168–0.170 with wz −0.023, and over 4 × 20 s it holds 0.164–0.166. It stands 4 × 20 s and
4 × 30 s with no falls (tilt 1.3°). The sweep gives 0.265 at 0.3 and 0.309 at 0.4, and `eval_velocity_env`
at 0.4 gives 0.343. Long stands were also run for 19800 (0 falls; at 0.2 for 20 s, two of four runs were
0.136 / 0.139, just under 0.14) and 19900 (0 falls; 0.159–0.161).

## Verdict
- **G3.3 confirmed on mac-m5.** Two or more consecutive checkpoints (three times) meet G1.3 with
  BAM (0.2 m/s within ±30 %, |wz| < 0.05, 0 falls), it stands ≥ 20 s, and the falls rule did not
  fire. Doubling the mirror loss steadied the yaw. Since MAC-31, |wz| at 0.2 has dropped to ≤ 0.05
  on most checkpoints.
- **WP3-18 not met.** Turn 1.0 reaches 0.47 at most, and ≥ 0.7 is needed. Turn 0.5 is about 0.
- Suggested reference: model_19700 (sha256 `398f399aab64332d13083a81a8b31ff96c06c40b2622b76a79eb30e424bb2846`,
  not committed).

Files: `mac32/ckpt-<it>/g1-3.json` for all 13 checkpoints. For the passing ones and 20000 there are
also `sweep-bam.json` and `turn-eval.json`. For 19700 / 19800 / 19900 there is `long-g13.json`, and for
19700 `eval.json`. Also `report.json` (history every 10th iteration), `run.json` (resume path
shortened) and `watch.log`.
