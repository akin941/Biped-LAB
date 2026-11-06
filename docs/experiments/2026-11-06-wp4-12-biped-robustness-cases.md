# G4.3 robustness cases for biped_v1: friction, pushes, latency (WP4-12)

```
env: cloud container (Linux, 4 vCPU), 2026-11-06
code: BipedVelocityEnv(action_delay=k), scripts/eval_biped.py --robustness
commands:
  python scripts/train_ppo.py --robot biped_v1 --envs 64 --iters 60 --workers 2 --seed 0 \
      --run-dir run0 --save-every 30 --out r0.json
  python scripts/eval_biped.py run0/policy.onnx --envs 4 --seconds 10 --robustness --out eval0.json
```

## Why
G4.3 asks for a robustness report covering pushes, friction and latency. Until now
`eval_biped.py --dr` only mixed pushes and friction into domain randomization and had
no latency at all, so a failure could not be traced to its cause.

## What was added
- `BipedVelocityEnv(action_delay=k)`: each policy action is applied `k` control periods
  (k x 20 ms) late; the observation's "last action" stays the policy's own output, as on
  the robot. Default 0 = training setting, unchanged (test: identical observations).
- `eval_biped.py --robustness`: seven extra cases at the 0.15 m/s command with DR off,
  one factor each:

| case | factor |
|---|---|
| `friction_0.5`, `friction_1.5` | foot friction fixed outside the DR range (0.7–1.3) |
| `push_forward`, `push_sideways` | one 0.3 m/s kick (1.5x the DR push) along the trunk heading / to its left at t = 4 s |
| `latency_20ms`, `latency_40ms`, `latency_60ms` | action delay of 1, 2, 3 control periods |

- The nightly `biped-ppo-smoke` job now runs the evaluation with `--robustness`
  (5 s, 2 envs) on every exported smoke policy.
- Tests: `tests/unit/test_biped_velocity_env.py` (delay applies actions k periods late,
  queue clears on reset, delay 0 unchanged), `tests/unit/test_eval_biped.py` (case list,
  a push loads the knees).

## Result (cloud, 60-iteration smoke policy, seed 0)
The pipeline runs end to end. The smoke policy only stands (as every policy at this
size so far, see WP4-11), so the numbers below check the plumbing, not G4.3:

| case | falls / 4 | vx_mean (m/s) | knee p95 (N·m) | slew-limited |
|---|---|---|---|---|
| nominal 0.15 m/s | 0 | 0.00 | 0.052 | 0.0 |
| friction 0.5 / 1.5 | 0 / 0 | 0.00 / 0.00 | 0.052 / 0.045 | 0.0 |
| push forward / sideways | 0 / 0 | 0.00 / 0.00 | 0.053 / 0.052 | 0.0 |
| latency 20 / 40 / 60 ms | 0 / 0 / 0 | 0.00 | 0.053 / 0.055 / 0.057 | 0.0 |

`vx_mean` is averaged over the run after a 2 s settle; a standing robot springs back
after a push, so it stays near 0 and the push shows up in the falls and torques.

## Against G4.3
Not evaluated here: G4.3 needs the full M5 run (MAC-08, `mac-m5`). The MAC-08 handoff
now asks for `--robustness` in its evaluation step, so the report covers all three
factors named in the gate.
