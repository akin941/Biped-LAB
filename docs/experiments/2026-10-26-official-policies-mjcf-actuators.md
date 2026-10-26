# Evidence: official policies in CPU MuJoCo with MJCF position actuators (WP1-07, part 1)

```
date: 2026-10-26
env: gha-linux (GitHub ubuntu-24.04), CI run 40298810061, job pretrained-policy
revision: PR #1 head (branch claude/project-thread-ukv8kb, commit "Establish Phase 0 foundation")
deps: mujoco 3.10.0, onnxruntime 1.24.4
policies: pollen-robotics/microduck-policies@088524a, sha256 verified
          alpha_walking e36332d3…, alpha_stand 15692687…
model: microduck_rl@b02fdea scene.xml (groundcontact), dt 0.005, decimation 4
actuator: MJCF <position> kp 0.55, ±0.96 N·m  (NOT the BAM model used in training)
command: uv run bipedlab eval-policy <policy> --out reports/<policy>.json
artifact: CI artifact "policy-reports"
contract tests: pytest -m policy -> 9 passed (both weights load as 61->14, finite rollouts)
```

| Policy | Scenario (10 s) | Fell | Max tilt | Mean vx (m/s) | Commanded |
|---|---|---|---|---|---|
| alpha_walking | stand_still | no | 0.61° | 0.00005 | 0 |
| alpha_walking | walk_forward_0.2 | no | 0.91° | **0.0009** | 0.2 |
| alpha_walking | walk_backward_0.15 | no | 0.83° | 0.00006 | −0.15 |
| alpha_walking | turn_left_0.5 | no | 0.82° | — (gyro z ≈ 0.001 rad/s) | wz 0.5 |
| alpha_stand | stand_still | no | 1.46° | 0.0002 | 0 |
| alpha_stand | (walk/turn cmds) | no | ≤ 2.2° | ≈ 0 | n/a for a stand policy |

## Interpretation

- **Balance transfers**: both policies keep the robot upright for 10 s in every scenario,
  whereas the same robot with no policy falls in 0.86 s. The 61/14 contract is right.
- **Locomotion does not transfer** with MJCF actuators: commanded 0.2 m/s, achieved
  < 0.001 m/s; no turning. G1.3 (vx within ±30 %) **fails** in this configuration.
- Likely cause (hypothesis, not yet tested): the policies were trained against the BAM
  XL330 model (voltage-limited torque, load-dependent friction, stiff friction solref),
  and the MJCF joint damping/frictionloss/armature differ. Testing this needs WP1-06.
- Other possible contributors to check: initial state (upstream starts at z = 0.125 m
  without settling; we settle onto the floor), runtime action scale/filters (WP1-09).
