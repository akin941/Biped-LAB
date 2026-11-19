# biped_v1 actuator sizing from our own gait (MAC-25 policy), M5 (MAC-35, WP4-25)

```
env: mac-m5 (macOS 27.0.1 arm64, 10 CPU, Python 3.12.13, MuJoCo 3.10.0), 2026-11-20
revision: c94e0d4 (main), git_dirty false; eval only, no training
policy: MAC-25 reference policy.onnx (sha256 3789a53b243f32d81492c42d0e8c793c31c10a35ddf26a3f9d15a4872745a52b)
run: eval_biped.py --dr --robustness --envs 8 --seconds 20 (sizing = 2x the worst walking max
     torque and p95 speed over the 6 nominal walking cases, DR off and on); size_biped.py (WP4-01)
```

## Per joint (both legs pooled)
| joint | walking max torque (N·m) | RMS (N·m) | p95 speed (rad/s) | **required, 2× (N·m)** | required speed (rad/s) | WP4-01 required (N·m / rad/s) |
|---|---|---|---|---|---|---|
| hip yaw | 0.186 | 0.054 | 1.42 | 0.37 | 2.8 | 0.21 / 2.2 |
| hip roll | **0.52 (clip)** | 0.183 | 2.76 | **≥ 1.04** | 5.5 | 0.35 / 2.5 |
| hip pitch | **0.52 (clip)** | 0.217 | 6.86 | **≥ 1.04** | **13.7** | 0.37 / 7.4 |
| knee | **0.52 (clip)** | 0.120 | 6.11 | **≥ 1.04** | **12.2** | 0.68 / 10.0 |
| ankle pitch | 0.404 | 0.120 | 3.86 | **0.81** | 7.7 | 0.44 / 5.3 |

At 0.25 m/s, hip pitch's p95 torque is also 0.52, so it is at the clip more than 5 % of the time (DR off and on).

## Robustness cases
No robustness case goes above the walking max. That is the expected outcome, though, because the
walking max is already the 0.52 N·m clip at hip roll, hip pitch and knee. push_sideways (knee and hip roll),
friction_1.5 and the pushes also reach the clip. The latency cases (20 / 40 / 60 ms) stay below it. 0 falls
in all 15 cases.

## Verdict
- **WP4-25 acceptance met**: the sizing block covers all five joint types, and the table above sits next to WP4-01.
- **Joints whose 2× requirement is above 0.52 N·m:** hip roll, hip pitch and knee (each ≥ 1.04 N·m) and ankle
  pitch (0.81). Only hip yaw (0.37) fits. On speed, hip pitch (13.7 rad/s) and knee (12.2 rad/s) also need
  more than the placeholder's 10 rad/s.
- Our own gait needs well above the WP4-01 estimate, which borrowed Microduck's dynamic factor: about 3× at
  hip roll and hip pitch, at least 1.5× at the knee, and 1.8× at the ankle. The XL330-class placeholder fails G4.1 at
  four of five joints.
- Caveat: three joints are torque-limited in the sim model, so their real demand is above what is measured
  here. To measure it, re-run with a higher `torque_limit` in the model, which feeds the servo choice in WP4-05.

Files: `mac35/eval.json` (with the `sizing` block and per-case `joints`; policy path shortened),
`mac35/size-wp4-01.json` (WP4-01 static sizing re-run).
