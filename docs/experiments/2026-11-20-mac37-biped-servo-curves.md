# biped_v1 with real servo torque–speed curves, M5 (MAC-37, WP4-28)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: dee26b3 (main), git_dirty false
curves: A = XC330-M288-T @ 6 V (--torque-limit 1.10 --motor-no-load-speed 10.2)
        B = STS3215 @ 7.4 V   (--torque-limit 1.91 --motor-no-load-speed 5.45)
baselines: MAC-25 policy.onnx under each curve (eval only)
fine-tunes: mac25/model_4000.pt + 1000 it with the MAC-25 flags and the curve (A 839 s, B 1258 s);
            falls rule not fired in either (falls ≤ 0.1 / it after it 100)
eval: eval_biped --dr --robustness --envs 8 --seconds 20 with the same curve
2× check: torque_max ≤ stall / 2 and speed_p95 ≤ no-load / 2 (A: 0.55 N·m, 5.10 rad/s; B: 0.955 N·m, 2.725 rad/s)
```

## MAC-25 rows (vx m/s / |wz| rad/s at 0.15 and 0.25; 0 falls in every case of all four runs)
| | DR off | DR on | 40 ms (of 0.15) | turn 0.8 | rows met? |
|---|---|---|---|---|---|
| baseline A (MAC-25) | 0.167 / 0.044, 0.247 / 0.016 | 0.143 / 0.055, 0.217 / 0.025 | 84 % | 0.580 | yes |
| **fine-tune A** | 0.156 / 0.069, 0.242 / **0.113** | 0.133 / 0.041, 0.211 / **0.088** | **76 %** | 0.892 | **no** |
| baseline B (MAC-25) | 0.158 / 0.061, 0.238 / 0.077 | 0.131 / 0.063, 0.206 / 0.080 | **76 %** | 0.581 | no |
| **fine-tune B** | 0.163 / 0.017, 0.244 / 0.005 | 0.142 / 0.009, 0.216 / 0.001 | 81 % | 0.873 | **yes** |

## Sizing and 2× check (fine-tunes; torque_max N·m / RMS / speed_p95 rad/s → pass, T = torque fails, S = speed fails)
| joint | A (XC330) | 2× A | B (STS3215) | 2× B |
|---|---|---|---|---|
| hip yaw | 0.249 / 0.057 / 2.03 | ✔ | 0.259 / 0.061 / 2.24 | ✔ |
| hip roll | 0.655 / 0.190 / 2.88 | T | 0.700 / 0.189 / 2.84 | S |
| hip pitch | 0.612 / 0.199 / 6.57 | T + S | 0.642 / 0.193 / 4.76 | S |
| knee | 0.456 / 0.114 / 6.19 | S | 0.464 / 0.122 / 4.79 | S |
| ankle pitch | 0.408 / 0.113 / 3.85 | ✔ | 0.391 / 0.104 / 3.67 | S |
| joints passing | | **2 / 5** | | **1 / 5** |

Baselines: MAC-25 under curve A passes 2 of 5 (hip yaw, ankle), and under curve B 1 of 5. `at_torque_limit` is
false everywhere.

## Verdict
- **B (STS3215 @ 7.4 V) can be learned.** The fine-tune meets every MAC-25 row under its own curve: |wz| ≤ 0.017,
  81 % at 40 ms, 0 falls, and it turns 0.87 at 0.8. Under B's curve the policy also slowed hip pitch and knee
  from about 6.5 to about 4.8 rad/s (p95). But B has the torque without the speed margin: every joint except hip
  yaw moves faster than half its no-load speed (2.7 rad/s).
- **A (XC330 @ 6 V)** is short on torque at the hips and on speed at hip pitch and knee. Its fine-tune drifts at
  0.25 (|wz| 0.113) and keeps only 76 % at 40 ms, so it does not meet the rows.
- Neither fine-tune passes more joints than MAC-25 does under curve A (2 of 5), so **MAC-25 stays the
  reference**.
- For WP4-05: at hip roll, hip pitch, knee and ankle, an actuator needs about 1 N·m of stall torque **and**
  at least about 5–10 rad/s no-load speed, for example a higher-voltage or lower-gear variant. B's torque with A's
  speed would pass.

ONNX sha256: A `d273bd75a02e6d91a709dfd7a3188d0f23d4c43e965397dfd1c747c733888c4b`,
B `13070c6af571cccaa7363e73ced8a2c0225d0e0e6082e658622af0e85d763bfb` (not committed).
Files: `mac37/baseline-a.json`, `mac37/baseline-b.json`, `mac37/{a,b}/` (`eval.json`, `run.json`,
`report.json` with history every 10th iteration, `watch.log`).
