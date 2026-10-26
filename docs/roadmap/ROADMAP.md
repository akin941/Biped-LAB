# Roadmap

Each phase has entry criteria, exit gate, and required evidence labels (ADR-0008).
Tasks are in [BACKLOG.md](BACKLOG.md). A phase gate is closed only when every
gate item has evidence in the named environment.

## Phase 0 — Research and foundation  *(status: exit gate pending CI)*
- **Entry:** empty repo.
- **Deliver:** verified research, ADRs, architecture, backlog, CI, security plan, handoff doc.
- **Exit gate:**
  - G0.1 research reports with V/U labels for 15 topics — `cloud` ✅
  - G0.2 CI green on main: static, unit, smoke on `gha-linux` and `gha-macos`, security scan
  - G0.3 owner can read README (Turkish) and know what is next

## Phase 1 — Existing Microduck simulation
- **Entry:** G0.2.
- **Exit gate:**
  - G1.1 official model loads and steps deterministically — `gha-linux`, `gha-macos`, `mac-m5`
  - G1.2 official `alpha_walking`/`alpha_stand` run through our 61/14 runner, contract verified — `gha-linux`
  - G1.3 with the BAM actuator, `alpha_stand` stands ≥ 20 s and `alpha_walking` tracks
    vx = 0.2 m/s within ±30 % over 10 s without falling — `gha-linux` and `mac-m5`
    - Status 2026-10-26 (`gha-linux`): **fails** with the official `alpha_walking`; its own
      upstream sim eval gives 0.4 → 0.17 m/s, which we reproduce (0.155). The gate stays as
      written; it needs a better policy (see `docs/experiments/2026-10-26-wp1-13-walking-command-response.md`).
  - G1.4 walking, turning, sit/stand, recovery behaviours characterized in a report — `gha-linux`
  - G1.5 interactive viewer of the policy on the M5 — `mac-m5`
  - G1.6 cross-check against the official HF Space visually (owner) — `mac-m5`

## Phase 2 — Observability
- **Entry:** G1.2.
- **Exit gate:**
  - G2.1 telemetry recorder: joints, IMU, commanded vs measured velocity, obs/actions,
    control period jitter, falls, actuator effort (Σ|τ·ω|), sim speed — `gha-linux`
  - G2.2 self-contained HTML/plot report per run, readable without code — `gha-linux`, reviewed by owner
  - G2.3 Turkish guide explaining every plot — owner sign-off

## Phase 3 — Reinforcement learning
- **Entry:** G1.3, G2.1.
- **Exit gate:**
  - G3.1 gymnasium env reproducing the 61/14 contract, validated by running the official
    policy inside it with the same metrics as G1.3 — `gha-linux`
  - G3.2 PPO smoke train (≤ 5 min) in CI-nightly; reward increases — `gha-linux`
  - G3.3 a policy trained from scratch on the M5 (CPU/MPS measured) stands and walks
    forward per G1.3 thresholds — `mac-m5`
  - G3.4 domain-randomization ablation report (with vs without) — `mac-m5`
  - G3.5 export to ONNX and run in the same runner — `mac-m5` + `gha-linux`
  - Every run records config, seed, git sha, metrics, artifact sha256.

## Phase 4 — Original biped in simulation
- **Entry:** G3.3.
- **Exit gate:**
  - G4.1 requirements + engineering calcs: mass budget, CoM, joint torque/speed from
    sim (peak & RMS with margin ≥ 2×), power, battery, thermal — reviewed by owner
  - G4.2 original MJCF (6–10 DoF, 20–30 cm), no third-party geometry — `gha-linux`
  - G4.3 trained policy walks in sim with DR; robustness report (pushes, friction, latency) — `mac-m5`
  - G4.4 Rust runtime drives the sim through a socket bridge at 50 Hz with the safety
    supervisor model — `gha-linux` + `mac-m5`
  - G4.5 BOM draft and actuator selection with cost — owner decision
- **Owner gate:** explicit approval required before any purchase.

## Phase 5 — Physical prototype
- **Entry:** Phase 4 gate + owner purchase approval.
- **Exit gate (all `hw`):** e-stop test, single-joint calibration, suspended motion,
  supported standing, first steps, sim-to-real comparison report.
  Simulation evidence never closes these.

## Phase 6 — Autonomous behaviours
- Obstacle detection, navigation, perception, voice, planner integration via the
  bounded command schema (ADR-0006). Detailed gates defined at Phase 5 exit.
