# WP4-04: Rust runtime skeleton, sim bridge and safety supervisor model

**Environment:** cloud (container, rustc 1.97) for the numbers below; CI job
`runtime` (gha-linux) repeats the tests and a 30 s real-time run on every push.
The M5 part of G4.4 is task MAC-08 in the Mac handoff. Simulation only: none of
this is hardware safety evidence (S-6).

## What exists
- `runtime/` crate `biped-runtime` (ADR-0007), std only, no third-party crates:
  - `protocol`: fixed-size little-endian frames over TCP (Hello, State, Action),
    with range checks on the header (joint count ≤ 64, finite ordered ranges).
  - `gate` (ADR-0006): command ranges (vx 0.3, vy 0.15 m/s, wz 1.0 rad/s), slew
    limits, 0.5 s freshness timeout → ramp to zero, non-finite commands dropped.
  - `controller`: `Stand` holds the stand pose (no policy for our robot until
    G4.3); `FaultInjector` makes NaN or jumping targets for tests only.
  - `safety`: the S-2 supervisor model. Latching faults → torque off: tilt > 45°,
    joint speed > 20 rad/s, non-finite state, state timeout (watchdog). Per
    period: non-finite targets → hold last safe targets; targets clamped 0.05 rad
    inside the joint range; slew limit 10 rad/s (servo class, WP4-03).
    Not modelled: temperature, bus voltage, current (no such sensors in sim).
  - binary: `--connect`, `--controller`, `--command START:END:VX:VY:WZ`,
    `--state-timeout-ms`, `--realtime`; one JSON summary line on exit.
- `bipedlab.runtime_bridge`: MuJoCo `biped_v1` as the robot, lockstep at 50 Hz
  (4 × 5 ms physics steps per period). Torque off = position gain 0, damping
  kept (limp servo). Push and stall injection for the tests.
- Tests: 9 Rust unit tests; `tests/unit/test_runtime_bridge.py` (Python stand-in
  client, runs everywhere); `tests/runtime/test_runtime_e2e.py` (needs the
  binary, marker `runtime`, required in the CI `runtime` job).

## Results (cloud)
| Scenario | Result |
|---|---|
| Stand 10 s, lockstep | 500/500 periods, no fault, max tilt 0.39°, runtime compute 15 µs mean / 141 µs max per period |
| Push 0.5 m/s sideways at 2 s | no fault, max tilt 7.1° |
| Push 1.5 m/s sideways at 2 s | tilt fault at the first state over 45° (step 109) and torque off in that same period |
| NaN targets from 1 s | 100 periods held at last safe targets, no fall, no latch |
| Left knee target +1 rad at 1 s | slew-limited to 0.2 rad per period over 5 periods; the open-loop stand then falls and the tilt fault fires |
| Sim stops sending states at 1 s | runtime watchdog after 200 ms (test) / 500 ms (default): sends torque off with the timeout bit, exits with code 3 |
| Real time, 5 s | 250 periods, 0 deadline misses, wall 5.04 s; worst period 23–32 ms (sleep jitter of the container) |

## Results (gha-linux, CI job `runtime`, PR #25 head e2db5ae)
- `pytest -m runtime`: 7 passed, with the same fault-step numbers as in the cloud
  (tilt fault and torque off at step 109 after the 1.5 m/s push; watchdog exit 3).
- 30 s wall-clock run at 50 Hz: 1500/1500 periods, **0 deadline misses**, worst
  period 20.18 ms, runtime compute 28.5 µs mean / 54 µs max, no fault.
- G4.4 gha-linux part met; the mac-m5 part is MAC-08.

## Known limits / follow-ups
- Only the feet of `biped_v1` collide (WP4-02), so a fallen robot sinks through
  the floor after torque off. Fine for the supervisor tests; add trunk/leg
  collision before fall-recovery work.
- The real-time pacing uses `thread::sleep`; worst-case period jitter is several
  ms on a shared container. A hardware loop needs a proper timer; measure on the
  SBC in Phase 5.
- ONNX inference (`ort`) comes with the first policy for our robot (G4.3), as its
  own reviewed dependency.
- Safety-layer code: owner review required before merge
  (docs/security/SECURITY_AND_SAFETY.md §2).
