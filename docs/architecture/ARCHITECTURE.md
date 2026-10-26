# Architecture proposal

Status: Phase 0 baseline (2026-10-26). Decisions referenced as ADR-xxxx.

## 1. System view (target, Phase 5+)

```
┌────────────────────── High-level compute (Linux SBC, optional) ─────────────────────┐
│ planner / LLM agent / perception (camera, ToF)        telemetry & logging           │
│        │ typed bounded commands (vx, vy, wz, posture, behaviour)                     │
│        ▼                                                                             │
│ command gate ──► locomotion policy (ONNX, 50 Hz) ──► joint targets                   │
└───────────────────────────────────────┬─────────────────────────────────────────────┘
                                        │ UART/CAN/SPI, framed + CRC, heartbeat
┌───────────────────────────────────────▼──────────── Low-level controller (MCU) ─────┐
│ safety supervisor: joint/vel/torque/temp/voltage limits, tilt, watchdog             │
│ servo bus driver, IMU read, state estimate                                          │
└───────────────────────────────────────┬─────────────────────────────────────────────┘
                                        ▼
                 servos ◄── motor power ◄── hardware e-stop (independent)
```
ADR-0006 governs the layering. In simulation, MuJoCo replaces the MCU + servos and
the same command → policy → actuator interfaces are kept.

## 2. Software components (repo)

| Component | Path | Phase | Language |
|---|---|---|---|
| Upstream pinning/fetch | `upstream.lock.toml`, `src/bipedlab/upstream.py` | 0 | Python |
| Model inspection | `src/bipedlab/sim/model.py`, `bipedlab inspect` | 0–1 | Python |
| Rollouts + stability metrics | `src/bipedlab/sim/rollout.py`, `bipedlab stand` | 1 | Python |
| Microduck policy runner (61/14 contract) | `src/bipedlab/sim/microduck_policy.py`, `bipedlab eval-policy` | 1 | Python |
| Pretrained policy provenance | `learning/policies/*.lock.toml`, `src/bipedlab/policies.py` | 1 | Python |
| BAM actuator port / integration | `src/bipedlab/sim/actuators/` (planned) | 1 | Python |
| Telemetry recorder + HTML report | `src/bipedlab/telemetry/` (planned) | 2 | Python |
| Training env (gymnasium) + PPO configs | `learning/training/` (planned) | 3 | Python |
| Evaluation harness | `learning/evaluation/` (planned) | 3 | Python |
| Custom biped models | `simulation/custom-biped/` | 4 | MJCF |
| Robot runtime (gate, controller, safety supervisor model) | `runtime/` crate `biped-runtime`; sim side `src/bipedlab/runtime_bridge.py` | 4–5 | Rust + Python |
| Firmware | `hardware/firmware/` (planned) | 5 | C/C++ or Rust |
| CAD, BOM, electronics | `hardware/` | 4–5 | — |

## 3. Repository layout

```
AGENTS.md  CLAUDE.md  README.md  LICENSE  NOTICE
upstream.lock.toml               pinned upstream shas
pyproject.toml  uv.lock          pinned Python env
src/bipedlab/                    our Python package (sim, policies, cli)
simulation/
  custom-biped/                  our own MJCF models (toy_biped_v0 = CI fixture)
  microduck/                     notes/configs for the upstream model (no meshes)
learning/
  policies/                      policy lockfiles (weights cached, gitignored)
  training/  evaluation/         Phase 3
runtime/                         Phase 4 (Rust crate biped-runtime, WP4-04)
hardware/                        Phase 4–5 (cad, electronics, bom)
tests/unit  tests/smoke          pytest tiers (see docs/testing)
scripts/                         fetch_upstream.py, fetch_policies.py
docs/
  architecture/ decisions/ research/ roadmap/ testing/ security/
  handoffs/ learning/ (Turkish) experiments/ (evidence)
third_party/                     gitignored pinned checkouts
```
Deviation from the brief's sketch: `roadmap/`, `testing/`, `security/`, `handoffs/`
added under `docs/`; runtime subfolders (`control/ safety/ telemetry/`) are created
when the Rust workspace exists, to avoid empty scaffolding.

## 4. Data flow in simulation (Phase 1)

```
pinned MJCF ─► MjModel (dt 0.005) ─► MjData
                    ▲                    │ sensors, qpos/qvel
                    │ ctrl = DEFAULT_POSE + action
              PolicyController ◄─ ObservationBuilder (61) ◄─┘
                    │ ONNX (CPU EP) action (14)
                    ▼
              RolloutResult + metrics (fall, tilt, speed) ─► JSON report
```

## 5. Interfaces to keep stable

- **Policy contract** `obs[1,61] float32 → actions[1,14]`; slices in
  `microduck_policy.OBS_SLICES`. Our own robot defines its own contract
  (likely smaller), versioned in a schema file.
- **Rollout report** JSON: environment label, versions, scenario metrics.
- **Command schema** (Phase 4): typed, bounded high-level commands (ADR-0006).
