# The runtime's +wz comes from its clean observation (MAC-45, WP4-37)

```
env: mac-m5 (macOS 27.0.1 arm64), 2026-11-23
revision: aed9352 (main), git_dirty false; eval only, no --realtime
run: eval_biped.py <MAC-42 average>/policy.onnx --envs 16 --seconds 20 --torque-limit 1.91 --motor-no-load-speed 5.45
     --obs-mode train | no-vel-lag | runtime; nominal rows, DR off; 0 falls in every case
```

| candidate | cmd | train vx / wz | no-vel-lag vx / wz | runtime vx / wz | MAC-44 runtime wz (8 starts) |
|---|---|---|---|---|---|
| k3_6900 | 0.15 | 0.164 / −0.063 | 0.172 / +0.057 | 0.173 / +0.065 | +0.076 |
| k3_6900 | 0.25 | 0.238 / −0.005 | 0.251 / +0.080 | 0.252 / +0.092 | +0.088 |
| k3_7450 | 0.15 | 0.182 / −0.068 | 0.191 / +0.022 | 0.192 / +0.006 | +0.021 |
| k3_7450 | 0.25 | 0.257 / +0.031 | 0.261 / +0.052 | 0.263 / +0.053 | +0.051 |
| k5_6950 | 0.15 | 0.169 / −0.027 | 0.177 / +0.071 | 0.179 / +0.065 | +0.089 |
| k5_6950 | 0.25 | 0.248 / +0.019 | 0.260 / +0.091 | 0.261 / +0.104 | +0.106 |
| k5_7250 | 0.15 | 0.176 / +0.009 | 0.187 / +0.056 | 0.186 / +0.076 | +0.084 |
| k5_7250 | 0.25 | 0.258 / +0.063 | 0.264 / +0.245 | 0.265 / +0.235 | +0.224 |

## Reading
- **The observation mismatch is the cause.** `runtime` mode reproduces the MAC-44 runtime wz within 0.024 in all 8 rows
  (6 of 8 within 0.015). `train` mode stays near 0 or goes negative.
- **The joint-velocity lag accounts for most of it.** `no-vel-lag` (noise on, current joint velocities) is within 0.02 of
  `runtime` in 7 of 8 rows. The actor was trained on one-period-old joint velocities, so fed current ones it turns right. Removing the
  noise and the gyro/gravity lag adds the rest. vx also rises about 5 %.
- The runtime path is faithful: the Python env with the runtime's observation gives the same result.
- **Next (cloud):** make training and runtime see the same observation. Either the runtime sends the joint velocities lagged
  as in training, or training drops the lag. Then re-score with `--obs-mode runtime`.

Files: `mac45/summary.json` (per candidate, mode and command: falls, vx_mean, wz_mean; plus the MAC-44 runtime wz).
