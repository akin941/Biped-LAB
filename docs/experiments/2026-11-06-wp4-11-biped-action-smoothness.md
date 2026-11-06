# WP4-11: smooth biped_v1 actions (slew limit in training, action-rate penalty)

**Environment:** cloud (container, 4 vCPU, x86_64, torch 2.15.0 from PyPI because the
PyTorch CPU index is blocked here; the nightly uses the pinned 2.9.1+cpu). Simulation
only: no evidence for hardware. Runtime numbers come from the Rust runtime of PR #25
(head 0b69732, built in a separate worktree, not part of this change).

## Problem
WP4-10 ran the 60-iteration PPO smoke policy in the Rust runtime: the supervisor's
10 rad/s target slew limit (0.2 rad per 20 ms period) fired in 488 of 500 periods and the
torque demand reached 1.37 × the servo limit. The policy jitters.

Cause, from the training logs: actions are scaled by 0.5 rad, but the actor started
with Microduck's exploration noise (std 1.0, made for scale 1.0). The action-rate penalty
on pure noise (~ −0.04 per step) outweighed what standing earned, so the smoke policies
fell within ~40 steps and the mean policy was never pushed to be smooth. The
action-rate weight (−0.1, ramping to −0.6 only after 1500 iterations) was too weak to
remove the jitter of the mean: after 300 iterations it gets worse (499 of 500 periods).

## Change
- `BipedVelocityEnv` applies the runtime supervisor's slew limit to the joint targets
  (`TARGET_SLEW` = 0.2 rad per step, same rule as `runtime/src/safety.rs`: step from
  the last applied target, clipped). The policy trains on what the robot executes. The
  policy still sees its own action as the last action, as in the runtime.
  `target_slew=None` restores the old behaviour.
- New reward term `action_slew` (weight −1.0): squared target step cut off by the limit.
- `ACTION_RATE_STAGES`: −0.5 from the start (was −0.1 → −0.6 over 1500 iterations).
- `train_ppo.py --init-std` (default 1.0 for Microduck, `BIPED_INIT_STD` = 0.5 for
  biped_v1; recorded in the report).
- Each step result has `slew_limited` (per env). `eval_biped.py` reports
  `slew_limited_fraction` and `wanted_target_step_mean` per case.
- The observation/action contract (39 → 10, `stand + 0.5 × action`) is unchanged, so
  the runtime of PR #25 needs no change.

## Results (cloud)
64 envs, 24 steps per env per iteration. "Runtime" = PR #25's `biped-runtime
--controller policy:` run 10 s (500 periods) through the sim bridge at command 0 and
at vx 0.15 m/s from 1 s; slew = periods with the slew-limit event. "Env" = this
branch's `eval_biped.py`, 8 envs × 10 s, command 0, observation noise on;
fraction of env-steps that hit the limit.

| Policy | Iter. | Runtime slew (cmd 0 / vx 0.15) | Runtime max torque demand | Env slew | Mean target step (rad) | Knee p95 (N·m) | Falls /8 (0.15, 0.25, 0, turn) |
|---|---|---|---|---|---|---|---|
| before, seed 0 | 60 | 488 / – | 1.37 | 69 % | 0.107 | 0.52 | 5, 8, 1, 2 |
| before, seed 1 | 60 | 490 / – | 1.55 | 93 % | 0.146 | 0.52 | 2, 1, 4, 4 |
| before, seed 0 | 300 | 498 / – | 2.81 | 99 % | 0.244 | 0.52 | 0, 0, 0, 1 |
| before, seed 1 | 300 | 500 / – | 2.71 | 100 % | 0.250 | 0.52 | 0, 0, 1, 1 |
| **after**, seed 0 | 60 | 0 / 0 | 0.70 | 0 % | 0.004 | 0.07 | 0, 0, 0, 0 |
| **after**, seed 0 ¹ | 300 | 0 / 0 | 0.69 | 0 % | 0.007 | 0.07 | 0, 0, 0, 0 |
| **after**, seed 1 | 300 | 0 / 0 | 0.80 | 0 % | 0.006 | 0.08 | 0, 0, 0, 0 |
| **after**, seed 2 | 300 | 0 / 0 | 1.00 | 0 % | 0.014 | 0.19 | 0, 0, 0, 0 |
| **after**, seed 0, DR | 300 | 0 / 0 | 0.95 | 0 % (also with DR in eval) | 0.009 | 0.13 | 0, 0, 0, 0 |

¹ Run with the same constants patched in at runtime before they were committed; the
other "after" rows run the committed `train_ppo.py` unchanged.

The 0.52 N·m knee p95 of the "before" policies is the servo force limit: the jitter
kept the knees saturated. The "after" policies ask for ~1/5 of it while standing.
No runtime run latched a fault. Per-step reward totals are not comparable across rows
(the penalty weights differ); the PPO smoke criterion (reward rises) still holds:
60 iterations seed 0 −0.0188 → −0.0021, 300 iterations −0.0083 → 0.0303 (seed 1).

### What mattered (seed 0, one factor at a time)
| Variant | Iter. | Env slew | Runtime slew |
|---|---|---|---|
| slew limit in env only (std 1.0) | 60 | 76 % | – |
| + `action_slew` −1 or −5 (std 1.0) | 60 | 29–43 %, **all envs fall** (penalties > reward: learns to fall) | – |
| std 0.5, no limit in env, no slew penalty | 60 | 48 % | – |
| std 0.3, no limit in env | 60 | 7 % | – |
| std 0.3, limit + `action_slew` | 60 | 0 % | – |
| std 0.3, limit + `action_slew`, action rate −0.1 | 300 | 54 % | 486 |
| std 0.3, limit + `action_slew`, action rate −0.5 | 300 | 0 % | 0 |
| std 0.5, limit + `action_slew`, action rate −0.5 (chosen) | 300 | 0 % | 0 |

So: lower exploration noise lets the policy learn to stand at all; the jitter only stays
away when the action-rate penalty is strong from the start. The env slew limit plus
`action_slew` make training match the runtime. std 0.5 was chosen over 0.3 to keep
more exploration for walking (0.25 rad per joint).

## Limits
- None of these policies walks yet (vx ≈ 0 in every case, before and after): 300
  iterations is ~0.5 M samples, MAC-08 is 3000 iterations at 256 envs. A stronger
  smoothness penalty can make standing still a local optimum; MAC-08 decides. If the
  M5 policy only stands, the next step is a ramp (e.g. −0.2 → −0.5 after 500
  iterations) or more exploration, not a weaker slew rule.
- The runtime numbers use PR #25's runtime (not merged); the nightly will repeat them
  on gha-linux once both are on `main`.
