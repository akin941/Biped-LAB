# Actuator survey against our own gait (WP4-27)

```
env: cloud (survey arithmetic only; the requirements come from mac-m5, MAC-36), 2026-11-20
command: uv run python scripts/actuator_survey.py --requirements mac36 --out docs/experiments/wp4-27/actuators-mac36.json
```

The requirements come from MAC-36, `GAIT_MAC36` in `bipedlab.design.actuators`. Both
the MAC-25 policy and its fine-tune were scored at a 1.5 N·m force limit, and no joint
reached that limit. Each value below is before the 2× margin.

| joint | peak torque N·m | p95 speed rad/s | RMS N·m | WP4-01 peak torque / speed |
|---|---|---|---|---|
| hip_yaw | 0.200 | 1.65 | 0.055 | 0.106 / 1.12 |
| hip_roll | 0.680 | 2.75 | 0.187 | 0.174 / 1.27 |
| hip_pitch | 0.585 | 7.10 | 0.172 | 0.182 / 3.69 |
| knee | 0.475 | 6.85 | 0.113 | 0.339 / 4.98 |
| ankle | 0.410 | 4.00 | 0.124 | 0.218 / 2.63 |

Candidates that pass (2× torque, 2× speed, loaded speed ≥ peak):
- hip_yaw: all eight
- hip_roll: XL430-W250-T @ 11.1 V only. STS3215 has the torque, but its speed margin
  is ×1.98.
- hip_pitch: **none**. The best are XC330-M288-T @ 6 V (×1.88 torque, ×1.43 speed) and
  STS3215 (×3.27 torque, ×0.77 speed).
- knee: **none**. The best is XC330-M288-T @ 6 V (×2.32 torque, ×1.48 speed).
- ankle: XC330-M288-T @ 5 V and @ 6 V

Verdict: speed is now the limit. The learned gait moves hip pitch and knee at
p95 ≈ 7 rad/s, and no surveyed hobby servo gives 2× that with the torque. The policy
was trained with an ideal position actuator, which has a force limit and no
torque-speed curve. So it has no reason to move slowly. Next is WP4-28: train with a
DC-motor torque-speed limit of a candidate, then size again. Nothing is bought, and
the WP4-05 choice stays with the owner.
