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

## Follow-up: upstream initial state (WP1-10)

```
env: gha-linux, CI run 40300195505, job pretrained-policy (PR #2)
change: start at z = 0.125 m, level, joints = DEFAULT_POSE, no settling (as upstream)
```

| alpha_walking, 10 s | init=upstream mean vx | init=settle mean vx |
|---|---|---|
| walk_forward_0.2 | 0.00098 m/s | 0.00087 m/s |
| walk_backward_0.15 | 0.00008 m/s | 0.00006 m/s |
| turn_left_0.5 (gyro z) | 0.0005 rad/s | 0.0010 rad/s |

No fall in any case. The start state is **not** the cause: locomotion still does not
emerge. Policy actions stay small (|a| ≤ 0.24 rad), i.e. the policy does not step.
Remaining prime suspect: actuator model (WP1-06, awaiting owner decision).

## Follow-up: telemetry of the forward-walk run (WP2-01)

```
env: gha-linux, CI job pretrained-policy (PR #3), alpha_walking, twist (0.2, 0, 0), 10 s
```

Peak joint torque 0.17 N·m (actuator limit 0.96), mean mechanical power 0.009 W,
foot contact fraction 0.994 left / 0.994 right, max |action| 0.24 rad. Both feet stay
on the ground almost the whole time: the policy shifts weight but never lifts a foot.

## Follow-up: what upstream's own simulator does (code reading, cloud)

At the pinned microduck_rl sha, `scripts/infer_policy.py` drives the joints through
`bam.mujoco.MujocoController` **by default**; MuJoCo position actuators are only the
`--no-bam` legacy path (lines 885-894, 1200, 1273-1281). Its action scale defaults to
1.0 and it applies no low-pass. So our runner equals upstream's legacy path, and the
upstream-supported way to run these policies in simulation uses BAM. This strengthens
the actuator hypothesis (still untested, WP1-06).

## Follow-up: robot-runtime action stage (WP1-09)

`PostProcess` reimplements the action stage of the robot daemon (`robotd/src/control.rs`):
target = DEFAULT_POSE + 0.9 · action, then low-pass α = 0.5 (head) / 0.7 (legs); the
observation still gets the raw action. The voltage term is 1.0 at the nominal 7.4 V and is
left off in the preset. CI compares `--post none` and `--post robotd-walking`:

```
env: gha-linux, CI run 40302566340, job pretrained-policy (PR #4), init=upstream
```

| alpha_walking, 10 s | post=none mean vx | post=robotd-walking mean vx |
|---|---|---|
| walk_forward_0.2 | 0.00098 m/s | 0.00090 m/s |
| walk_backward_0.15 | 0.00008 m/s | 0.00013 m/s |
| turn_left_0.5 (gyro z) | 0.0005 rad/s | -0.0014 rad/s |

Telemetry, forward 0.2 m/s: peak torque 0.169 vs 0.154 N·m, mean power 0.0092 vs
0.0090 W. No fall in any case. The runtime action stage is **not** the cause either;
it only damps the (already small) actions. Remaining prime suspect: actuator model
(WP1-06, awaiting owner decision).
