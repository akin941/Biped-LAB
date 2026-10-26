# ADR-0010: BAM XL330 actuator as a reviewed in-repo port, not an installed package

Status: accepted (2026-10-26)

## Context
The official Microduck policies are trained against the BAM M6 model of the
Dynamixel XL330 (Rhoban/bam, branch `mjlab_frictionloss`, commit
`62bd8ce12154340be97e06f7f41a0ca8f116d967`, Apache-2.0). With MuJoCo position
actuators the walking policy balances but does not walk (WP1-07). The owner
delegated the choice to the agents; the recommendation was to depend on BAM pinned
to the commit microduck_rl uses.

Review of that commit (cloud, 2026-10-26): Apache-2.0; core dependencies only
`numpy` and `colorama`; the modules upstream's CPU simulator uses (`model.py`,
`actuator.py`, `dynamixel/actuator.py`, `mujoco.py`, `params/xl330/m6.json`) are
pure numerics with no I/O, network, subprocess or eval on the import path.
Installing or running the package from git in the cloud session was refused by
the session's safety policy for code from external sources.

## Decision
- Port the ~150 lines upstream actually uses into `src/bipedlab/sim/bam_xl330.py`:
  firmware P-loop to PWM to volts, DC motor torque with back-EMF, the M6 friction
  budget, the per-physics-step controller, and the spec edits from
  `infer_policy.load_mujoco_with_bam`. Parameters are copied from `m6.json`.
- Header cites the source commit; `NOTICE` carries the Apache-2.0 attribution.
- Unit tests check each equation against hand-computed values. Where upstream's CPU
  controller has a quirk (efc_id compared with joint ids), the port keeps it and
  says so, because the goal is to reproduce upstream's simulator.
- Runner option `actuator="bam"` (CLI `--actuator bam`); every report names its actuator.
- Revisit installing the pinned package if a later need (e.g. other BAM models or
  GPU training via mjlab) makes a port impractical; that would be its own reviewed PR.

## Consequences
No third-party code runs in our process; the port must be re-checked when the
upstream pin moves. Residual risk: a porting mistake. Mitigation: hand-computed
unit tests, and a cross-check that the linearised BAM gain (0.5545 N·m/rad) and
force limit (0.963 N·m) reproduce the MJCF actuator constants (kp 0.55, ±0.96).
