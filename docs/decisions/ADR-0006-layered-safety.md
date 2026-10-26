# ADR-0006: Layered control; AI never commands motors directly

Status: accepted (2026-10-26)

## Context
Phase 6 may add LLM/agent planners. The Microduck runtime has thin safety (±π clamp,
500 ms command timeout, no per-joint limits, no e-stop). Our robot must be safer.

## Decision
```
AI planner / LLM / perception      (non-real-time, untrusted output)
      │  bounded high-level commands: vx, vy, wz, posture, behaviour id
      ▼
Command gate                       (rate limits, ranges, timeout → zero)
      ▼
Locomotion policy (ONNX, 50 Hz)    (joint targets)
      ▼
Safety supervisor (MCU, ≥ 200 Hz)  (joint/vel/torque/temp/voltage limits, tilt, watchdog)
      ▼
Motor drivers ── hardware e-stop cuts motor power independently of all software
```
- An LLM may only emit commands from a typed, bounded schema; never joint targets.
- The safety supervisor runs on separate hardware from the high-level computer.
- Hardware e-stop is mandatory before first motion.

## Consequences
Simulation code already separates command → policy → actuator, so the same
interfaces carry to hardware. Safety requirements live in
`docs/security/SECURITY_AND_SAFETY.md`.
