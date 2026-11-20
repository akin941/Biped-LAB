# biped_v1 on an uneven floor, M5 (MAC-39, WP4-31)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0), 2026-11-20
revision: b7bc8d6 (main), git_dirty false; eval only
policies: STS3215 fine-tune (MAC-37 B, sha256 13070c6a…63bfb) under its curve
          (--torque-limit 1.91 --motor-no-load-speed 5.45); MAC-25 reference (sha256 3789a53b…a52b)
          at the default 0.52 N·m
run: eval_biped.py --terrain (random height field, heights uniform in [−h, 0] every 10 cm, default seed),
     0.15 m/s command, DR off, 16 envs × 20 s; the STS3215 run also with --robustness
```

## Uneven floor (0.15 m/s; falls of 16; vx / wz averaged over envs still standing)
| h | STS3215: falls / vx / wz | MAC-25: falls / vx / wz |
|---|---|---|
| 0 (flat) | 0 / 0.163 / +0.016 | 0 / 0.166 / −0.046 |
| 5 mm | **3** / 0.153 / −0.018 | **4** / 0.156 / −0.041 |
| 10 mm | **14** / 0.114 / −0.038 | **12** / 0.117 / −0.061 |
| 20 mm | **16** / −0.053 / −0.117 | **16** / −0.050 / −0.184 |

STS3215 on the flat floor (robustness rows, 16 envs) had 0 falls in every case: nominal 0.163 / 0.244, friction 0.5 / 1.5
gave 0.162 / 0.171, pushes 0.164 / 0.163, and latency 20 / 40 / 60 ms gave 0.141 / 0.121 / 0.106. Turn 0.8 reached wz 0.873.

## Verdict
- **WP4-31 not met.** At `rough_10mm` the STS3215 policy falls in 14 of 16 envs (0 needed). It **falls**,
  not just slows down. `vx_mean` only counts the envs still standing, so its 0.114 comes from the walking before
  each fall and from the two survivors. A 5 mm floor already causes 3 of 16 falls, and at 20 mm both policies
  stumble backwards and all fall.
- MAC-25 does no better (4 / 12 / 16 falls), so the STS3215 fine-tune is not the cause. Neither policy has
  trained on anything but a flat plane.
- Next (cloud): train with the height field as a terrain curriculum (0 up to 10–20 mm), keeping the flat-floor
  rows, then repeat this check.

Files: `mac39/sts3215.json`, `mac39/mac25.json` (policy paths shortened).
