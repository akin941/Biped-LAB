# biped_v1 torque demand without the 0.52 N·m clip, M5 (MAC-36, WP4-26)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0, torch 2.9.1), 2026-11-20
revision: c0836f2 (main), git_dirty false
step 1: MAC-25 policy.onnx (sha256 3789a53b…a52b), eval_biped --dr --robustness --envs 8 --seconds 20
        --torque-limit 1.5 (no training)
step 2: MAC-25 model_4000.pt + 500 it with the MAC-25 flags and --torque-limit 1.5 (576 s, falls rule
        not fired, falls ≤ 0.14 / it after it 100), then the same eval (sha256 e992476a…7ac3)
```

## Required torque / speed per joint (2 × walking max torque, 2 × p95 speed; both legs pooled)
| joint | MAC-35 (limit 0.52) | step 1: MAC-25 at 1.5 | step 2: fine-tune at 1.5 | RMS (step 1 / 2) |
|---|---|---|---|---|
| hip yaw | 0.37 / 2.8 | 0.39 / 3.0 | 0.40 / 3.3 | 0.055 / 0.049 |
| hip roll | ≥ 1.04 (clip) / 5.5 | **1.19** / 5.5 | **1.36** / 5.2 | 0.180 / 0.187 |
| hip pitch | ≥ 1.04 (clip) / 13.7 | **1.17** / **14.2** | **1.14** / **14.1** | 0.172 / 0.167 |
| knee | ≥ 1.04 (clip) / 12.2 | 0.82 / **12.1** | 0.95 / **13.7** | 0.109 / 0.113 |
| ankle pitch | 0.81 / 7.7 | 0.76 / 8.0 | 0.82 / 8.0 | 0.119 / 0.124 |

(N·m / rad/s.) `at_torque_limit` is false for every joint in both steps, so no joint reaches 1.5 and these are
measured requirements. The highest walking maxima were hip roll 0.679 N·m (step 2, DR on, 0.25 m/s) and
hip pitch 0.583 (step 1). The knee does not need its MAC-35 lower bound: with headroom it peaks at
0.41–0.47 N·m. No robustness case goes above the walking max; the highest are friction_1.5 at hip roll 0.61
and push_sideways at hip roll 0.59.

## Step 2 against the MAC-25 rows
| row | step 2 | MAC-25 | pass? |
|---|---|---|---|
| \|wz\| ≤ 0.08 at 0.15 / 0.25, DR off | 0.034 / 0.032 (vx 0.161 / 0.252) | 0.047 / 0.006 | yes |
| \|wz\| ≤ 0.08 at 0.15 / 0.25, DR on | 0.048 / 0.010 (vx 0.132 / 0.215) | 0.052 / 0.028 | yes |
| 40 ms latency ≥ 80 % of vx | **73 %** (0.110) | 83 % | **no** |
| 0 falls | 0 | 0 | yes |
| (zero command \|wz\|, turn 0.8) | 0.098 / 0.056, 0.717 / 0.626 | 0.034 / 0.020, 0.579 / 0.523 | (not a MAC-36 row) |

Step 1 (the MAC-25 gait itself, run at the 1.5 limit) keeps 84 % at 40 ms and shows 0 falls in all 15 cases.

## Verdict
- **Demand is measured, not a lower bound:** no joint reaches 1.5 in either step. With the 2× margin,
  the joints need: hip roll 1.2–1.4 N·m, hip pitch about 1.15 N·m at about 14 rad/s, knee 0.8–0.95 N·m at
  12–14 rad/s, ankle about 0.8 N·m, and hip yaw about 0.4 N·m. Steps 1 and 2 agree within about 15 %.
- **WP4-26 only partly met.** No `at_torque_limit` (met), but the step 2 policy fails the 40 ms latency row
  (73 %), and its zero-command yaw got worse. MAC-25 stays the reference.
- For WP4-05, the servo must give about 1.2–1.4 N·m at the hips and handle about 14 rad/s at hip pitch and
  knee. That rules out the XL330 class (0.52 N·m, 10 rad/s) at four of five joints.

Files: `mac36/eval-mac25-tl1.5.json` (step 1), `mac36/eval.json` (step 2), `mac36/report.json` (history
every 10th iteration), `mac36/run.json`, `mac36/watch.log` (local paths shortened).
