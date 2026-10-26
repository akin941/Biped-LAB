# BAM start-state bug, alpha_stand under BAM, and re-measured BAM results (WP1-12)

```
env: gha-linux, CI run 40311668607, job pretrained-policy (PR #12)
init: upstream (z = 0.125 m, joints at DEFAULT_POSE, no settle); 10 s per scenario
bam: our port of Rhoban/bam@62bd8ce (ADR-0010), kp_fw 200, vin 7.4 V, drop gain 0.1
repeats: where marked, 1 run from the upstream start + perturbed starts (< 1 encoder count)
```

## The bug

`bam_xl330.Controller.reset()` called `mujoco.mj_setConst(model, data)` on the rollout's
data, after the start state was set. `mj_setConst` writes `qpos0` into the data it is
given, so every BAM rollout before this fix started with **all joints at 0** (straight
legs, head level) and the trunk at 0.12 m. In the kick scenes the ball was also moved
back. Upstream BAM calls `mj_setConst` once, in the `MujocoController` constructor,
before any state is set. Found while reading the telemetry profile of an alpha_stand
run (joints read 0 at the first step); confirmed in the cloud with a synthetic policy.

Fix: `mj_setConst` runs on scratch data in `Controller.__post_init__`; `reset()` only
sets the target. Regression test: `test_reset_keeps_the_start_state`. Also, `data.ctrl`
is now 0 for BAM motors at the upstream start (as upstream: BAM's ctrl is torque, the
position target goes to the controller), which removes a spurious 0.46 N·m first-step
torque from telemetry.

All BAM results in `2026-10-26-bam-vs-mjcf-actuators.md` and
`2026-10-26-wp1-11-training-time-factors.md` were measured with the bug and are
superseded by the numbers below.

## alpha_stand (the WP1-12 question)

alpha_stand is upstream's get-up / sit-to-stand teacher (wandb `69u48n8l` @ 9750,
`microduck_rl` `docs/velstand_policy.md`). It never saw a twist command (obs normaliser
std 0.005, `distill.py`), so it is now evaluated with `--scenarios standing`: zero twist,
body pitch/roll commands in the body-pose slots. Trunk angles: mean over the second half,
from the projected-gravity observation. 4 runs per row (BAM); no fall in any run.

| scenario | MJCF | BAM | BAM kp 160 | BAM delay 4 + vel lag 1 |
|---|---|---|---|---|
| stand_still: max tilt | 1.4° | 1.8° | 2.1° | 2.1° |
| body pitch +10°: trunk pitch | – | +9.6° | +9.8° | +9.3° |
| body pitch −10°: trunk pitch | – | −9.7° | −9.8° | −9.6° |
| body roll +10°: trunk roll | – | +9.0° | +8.9° | +9.1° |

Spread over repeats ≤ 0.03°. `--init settle` gives the same. Upstream reports 9.2–9.5° for
a 10° command on this policy; ours is 9.0–9.7°. **The 46–53° lean was the start-state
bug**: with the fix alpha_stand stands within 2° and tracks body tilt as documented, with
the training gain (kp 200) and with robotd's standing gain (kp 160). Fed the walking
twists (out of distribution), it still just stands (max tilt ≤ 2.0°).

## alpha_walking, re-measured

| scenario | MJCF vx / yaw | BAM vx / yaw (8 runs) |
|---|---|---|
| walk_forward_0.2 | 0.001 m/s | 0.001 ± 0.000 m/s |
| walk_backward_0.15 | 0.000 m/s | 0.000 ± 0.000 m/s |
| turn_left_0.5 | 0.000 rad/s | 0.001 ± 0.000 rad/s |

Telemetry at forward 0.2 m/s: feet in contact 99.4 % of steps with both actuators; max
|action| 0.24 rad. **With the correct start state the policy does not step with either
actuator**: it stands. The WP1-11 factors (delay 3/6, joint_vel lag, 6.5/8.2 V, delay 4
+ lag 1; 8 runs each) change nothing: forward vx 0.001 m/s in every row, no fall.

The gait reported earlier with BAM came from the wrong start: dropped from straight legs,
the robot stumbled and the policy kept stepping. That it can walk once stepping is a
hint, not a result. **G1.3 fails**: alpha_stand stands (20 s not yet run as such; 10 s
scenarios here), alpha_walking does not track 0.2 m/s. Why the policy does not start a
gait from the upstream start is WP1-13.

## Behaviours, re-measured

| behaviour | MJCF | BAM |
|---|---|---|
| sitstand | sits (5.9 cm), stands back (11.6 cm) | same (5.9 cm → 11.6 cm) |
| ground_pick | crouch to 8.4 cm, upright | crouch to 8.7 cm, upright |
| kick_left | ball moved 0.97 m | ball moved 0.57 m |
| kick_right | ball did not move | **ball moved 1.52 m** |
| roulade | ends on its side (94°) | **full roll (178°), ends upright (1.2°)** |

With BAM all five behaviour policies do what they are for and end upright.

## Reproduce

```
uv run bipedlab eval-policy alpha_stand.onnx --scenarios standing --actuator bam --repeats 4
uv run bipedlab eval-policy alpha_walking.onnx --actuator bam --repeats 8
uv run bipedlab eval-behaviours --actuator bam
uv run bipedlab record alpha_stand.onnx --actuator bam --out runs/stand && uv run bipedlab profile runs/stand.npz
```
