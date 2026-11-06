# WP4-10: biped_v1 policy as a runtime controller (G4.3 preparation)

**Environment:** cloud (container, 4 vCPU, x86_64) for every result below. The
nightly `biped-ppo-smoke` job repeats the trained-policy part on gha-linux. Simulation
only: this is no evidence for hardware. Stacked on WP4-04/08/09 (PR #25).

## Change
- `runtime/src/policy.rs`: `Policy` controller. Builds the biped_v1 actor
  observation (gyro | projected gravity | q − stand | q̇ | last action | command,
  39 floats) from the bridge state and the gated command, runs the MLP (ADR-0013)
  and returns `stand + 0.5 × action`. Rejects a policy whose size does not fit the
  robot (`3n + 9 → n`). A non-finite action becomes NaN targets (the supervisor
  holds the last safe targets) and is not fed back as the last action.
- `biped-runtime --controller policy:FILE [--action-scale 0.5]`.
- `bipedlab.runtime_bridge`: `projected_gravity`, `policy_observation` (Python
  mirror of the Rust builder), `BridgeConfig.record` (per-period trace), and
  `run_with_runtime` now returns the runtime's stderr when it quits early.
- `scripts/runtime_sim.py --require-clean-exit`.
- Nightly `biped-ppo-smoke`: export the trained ONNX to the runtime format with the
  parity check, then run it 10 s in the runtime loop.

## Results (cloud)
- Contract: `policy_observation` built from what the bridge sends (IMU site sensors,
  joint state, keyframe stand pose) equals the clean actor terms of
  `BipedVelocityEnv` (first 39 critic entries) within 1e-6 over 5 random steps.
- Runtime vs Python reference, random small 39 → 64 → 64 → 10 ELU actor, 3 s
  (150 periods, no supervisor event): targets equal within 1e-5.
- A policy with large outputs is slew-limited (≤ 0.2 rad per period), not obeyed.
- A Microduck-sized policy (61 → 14) is refused: exit 2, "policy is 61 -> 14, the
  robot needs 39 -> 10".
- Rust unit tests: 17 pass (4 new in `policy.rs`).

### Trained policy end to end (cloud)
PPO smoke on biped_v1, seed 0, 64 envs × 60 iterations (38 s; reward per step
−0.0155 → −0.0062). Torch here was 2.15.0 from PyPI because the PyTorch CPU index
is blocked in the cloud; the nightly uses the pinned 2.9.1+cpu.

| Step | Result |
|---|---|
| rsl-rl ONNX graph | Sub, Div, Gemm, Elu ×3, Gemm (input normaliser + MLP): all supported |
| export parity | numpy 1.9e-6, Rust 1.4e-6 max abs error vs onnxruntime |
| runtime, 10 s, command 0 | 500 periods, max tilt 3.2°, no fault; slew limit active in 488 periods |
| replay (Python reference + 10 rad/s slew model) | equals the runtime targets within 1.2e-7 over all 500 periods |
| runtime, 10 s, vx 0.15 m/s from 1 s | 500 periods, max tilt 3.2°, no fault (the smoke policy does not walk) |
| runtime, 30 s real time | 1500 periods, 0 deadline misses, worst period 21.8 ms, mean compute 145 µs |

Observation: the 60-iteration smoke policy jitters (it asks for steps > 0.2 rad per
period almost every period) and its torque demand reaches 1.37 × the servo limit
without `--torque-fraction`. The supervisor caps the steps; the full M5 training
(MAC-08) should add an action-rate penalty strong enough that the slew limit stays
idle, and its policy should be run here with `--torque-fraction` before any
hardware discussion.

## Results (gha-linux)
Run 42170283487 (`nightly-train` on PR #25, head d951cf5): all four `biped-ppo-smoke`
jobs (seeds 0, 1, 2 and seed 0 with DR) trained, exported their policy with the
Rust parity check passing, and ran it 10 s (500 periods) in the runtime with a clean
exit. Per-run JSON is in the job artifacts (`reports/biped/runtime_*.json`).

## Gate status
G4.3 is not met: it needs the MAC-08 policy. This removes the integration risk
between training and the runtime: an rsl-rl biped_v1 policy runs in the runtime
with the same observation and action as in training.
