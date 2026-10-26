# ADR-0007: Rust only for the on-robot runtime, adopted in Phase 4

Status: accepted (2026-10-26)

## Context
Microduck's runtime is a mature Rust workspace (ort, JSON-RPC daemons, signed
updates). Rust gives memory safety and predictable latency on an SBC. But Phases 1–3
are simulation/RL work where Python dominates, and the owner is learning.

## Decision
- Phases 1–3: Python only. Study Microduck's Rust design as reading material
  (`docs/learning/`).
- Phase 4: introduce a Rust crate for the robot-side control loop (policy inference via
  `ort`, command gate, telemetry), first exercised against our MuJoCo sim through a
  socket bridge (sim-in-the-loop), then on the SBC.
- Firmware on the low-level controller (e.g. ESP32-S3) in C/C++ or Rust (embedded),
  decided by a later ADR.

## Consequences
No Rust toolchain in CI until Phase 4. Avoids adopting Rust "because it is popular".
