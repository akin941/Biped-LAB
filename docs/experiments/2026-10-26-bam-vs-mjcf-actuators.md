# BAM XL330 M6 vs MJCF position actuators (WP1-06)

> **Correction (2026-10-26, WP1-12).** Every BAM number in this report was measured
> with a bug: `bam_xl330.Controller.reset()` called `mj_setConst` on the rollout's data,
> which overwrote the start state with `qpos0` (all joints 0, trunk 0.12 m; in the kick
> scenes the ball too). Upstream calls it once, before the state is set. Fixed in the
> WP1-12 PR; the BAM results are re-measured there and this report is superseded where
> they differ.

```
env: gha-linux, CI run 40306731638, job pretrained-policy (PR #10)
init: upstream (z = 0.125 m, no settle); 10 s per walking scenario
bam: our port of Rhoban/bam@62bd8ce (ADR-0010), kp_fw 200, vin 7.4 V, drop gain 0.1
```

## Walking (alpha_walking, scene.xml)

Single runs; see the WP1-11 report for spread over perturbed starts.

| scenario | MJCF mean vx / yaw rate | BAM mean vx / yaw rate | BAM max tilt |
|---|---|---|---|
| stand_still | 0.0002 m/s | 0.008 m/s | 6.4° |
| walk_forward_0.2 | 0.0010 m/s | **0.062 m/s** | 5.3° |
| walk_backward_0.15 | 0.0001 m/s | **−0.027 m/s** | 4.6° |
| turn_left_0.5 (gyro z) | 0.000 rad/s | **0.32 rad/s** | 4.5° |

Telemetry, forward 0.2 m/s: foot contact 0.52 / 0.52 with BAM (alternating steps)
vs 0.99 / 0.99 with MJCF; peak torque 0.64 vs 0.17 N·m; mean mechanical power 0.81 vs
0.009 W; max |action| 1.44 vs 0.24 rad. No fall in any case.

**The robot walks with BAM.** It steps, moves in the commanded direction and turns the
commanded way. Tracking is still short: 31 % of the forward command, 18 % of backward,
64 % of the turn command. **G1.3 (vx within ±30 % of 0.2 m/s) still fails**, now by
amount instead of by absence of gait.

`scene_walk.xml` (the collision model walking is trained on) gives results identical to
`scene.xml` with both actuators: in these runs the leg meshes never touch the floor, so
the extra collision geoms do not matter here.

## Behaviours

| behaviour | MJCF | BAM |
|---|---|---|
| sitstand | sits (5.9 cm), stands back | same (5.7 cm → 11.6 cm), max tilt 31° |
| ground_pick | crouch to 8.4 cm, upright | crouch to 8.7 cm, upright |
| kick_left | ball moved 0.97 m | ball moved 0.21 m |
| kick_right | ball did not move | **ball moved 0.21 m** |
| roulade | ends on its side (94°) | **full roll (178°), ends upright (1.4°)** |

## alpha_stand with BAM: open question

alpha_stand stays up (no fall) but leans 46–53° in three of four scenarios, with actions
up to 7.6 rad; with MJCF it stood within 2°. Not explained yet. Candidates: the start
state, or the standing preset in robotd (softer gain: `standing_gain_ratio = 0.8`, i.e.
kp_fw 160; action scale 1.0), which we do not apply. To investigate (WP1-12).

## Interpretation

The actuator model was the main reason walking failed: with BAM's voltage-limited motor
and load-dependent friction the same policy produces a gait. The remaining tracking gap
has known, untested candidates from the training config
(`microduck_velocity_env_cfg.py`, `microduck_constants.py` at the pinned sha):
- action delay of 3–6 physics steps (15–30 ms) on the actuator during training; none here;
- one-control-step lag on the joint_vel observation during training; none here;
- battery voltage randomized 6.5–8.2 V in training; fixed 7.4 V here.
These are WP1-11; results in `2026-10-26-wp1-11-training-time-factors.md` (delay helps
most, best 45 % of the forward command, G1.3 still fails). BAM numbers above are single
runs and vary by a few percent between CI runner CPUs; that report uses 8 runs per row.
