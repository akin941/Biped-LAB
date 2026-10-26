# Backlog

Status: `done` · `ready` · `blocked` · `awaiting-mac` · `awaiting-owner` · `later`.
Env: where the task must run (`cloud` includes GitHub Linux CI). Pick the top `ready`
task matching your environment whose deps are `done`.

## Phase 0

| ID | Task | Env | Deps | Status | Acceptance criteria | Evidence |
|---|---|---|---|---|---|---|
| WP0-01 | Research upstream repos at pinned shas | cloud | – | done | 15 topics, V/U labels, line citations | `docs/research/` |
| WP0-02 | ADRs, architecture, tech matrix, risks, security plan | cloud | WP0-01 | done | files exist, cross-linked | `docs/` |
| WP0-03 | Python package skeleton, uv.lock, ruff/mypy strict | cloud | – | done | `ruff check`, `ruff format --check`, `mypy` clean | PR #1 |
| WP0-04 | CI: static, unit, smoke (linux + macos arm64), security | cloud | WP0-03 | done | all jobs green on PR and main | PR #1 checks (7/7 green, incl. macos-15) |
| WP0-05 | Mac handoff doc + Turkish README/primer | cloud | WP0-02 | done | MAC tasks have commands, expected output, criteria | `docs/handoffs/` |
| WP0-06 | Branch protection on `main` requiring CI | owner | WP0-04 | awaiting-owner | settings show required checks | owner |

## Phase 1

| ID | Task | Env | Deps | Status | Acceptance criteria | Evidence |
|---|---|---|---|---|---|---|
| WP1-01 | Pinned upstream fetch with sha verification | cloud | WP0-03 | done | rejects branch names / non-github URLs; HEAD == pin | `tests/unit/test_upstream_lock.py` |
| WP1-02 | Model inspector (`bipedlab inspect`) | cloud | WP1-01 | done | lists 14 actuators, gains, keyframes for Microduck | `tests/smoke/test_microduck_model.py` |
| WP1-03 | Open-loop stand + determinism characterization | cloud | WP1-02 | done | same digest twice; Microduck falls < 2 s open-loop; toy stands 5 s | `docs/experiments/2026-10-26-phase1-cloud-baseline.md` |
| WP1-04 | 61/14 observation builder + ONNX runner | cloud | WP1-02 | done | obs slices verified (gravity, joint rel, command); bad shapes / NaN rejected | `tests/smoke/test_microduck_policy.py` |
| WP1-05 | Pretrained policy lock + sha256-verified fetch | cloud | – | done (fetch runs in gha-linux; HF blocked in cloud container) | tampered file rejected; CI downloads and verifies | `tests/unit/test_policy_lock.py`, CI `pretrained-policy` |
| WP1-06 | BAM actuator integration (pinned Rhoban/bam) for CPU MuJoCo | cloud | WP1-04 | awaiting-owner (adds third-party code: approve pinned dependency or own port) | XL330 M6 model loads; unit test vs a hand-computed torque; runner option `actuator=bam` | – |
| WP1-07 | Official policy behaviour report (stand/walk/turn) with MJCF vs BAM actuators | cloud | WP1-05, WP1-06 | MJCF part done; BAM part blocked on WP1-06 | report JSON+MD in `docs/experiments/`; G1.3 thresholds evaluated (pass or documented fail) | `docs/experiments/2026-10-26-official-policies-mjcf-actuators.md` (G1.3 fails with MJCF actuators) |
| WP1-08 | Sit/stand, ground pick, kick, roulade scenario support (phase commands) | cloud | WP1-07 | ready | each policy runs its documented command; report per behaviour | – |
| WP1-10 | Match upstream initial state (z = 0.125 m, no settle) and test its effect on walking | cloud | WP1-07 | done | report compares settle vs upstream init | experiments/2026-10-26-official-policies-mjcf-actuators.md (no effect) |
| WP1-09 | Runtime post-processing model (action scale 0.9, voltage comp, low-pass) as optional runner stage | cloud | WP1-07 | in-review | toggleable; report shows effect | `PostProcess` + `--post robotd-walking`; CI report `alpha_walking_robotd.json` |
| MAC-01..04 | Native Mac setup, tests, viewer, benchmark | mac-m5 | WP0-04 | awaiting-mac | see handoff | `docs/handoffs/MAC_LOCAL_HANDOFF.md` |

## Phase 2

| ID | Task | Env | Deps | Status | Acceptance criteria |
|---|---|---|---|---|---|
| WP2-01 | Telemetry recorder (npz/parquet) for rollouts | cloud | WP1-04 | done (`bipedlab record`, `src/bipedlab/telemetry/`) | records all G2.1 signals at 50 Hz; schema documented; size < 5 MB per 10 s |
| WP2-02 | Static HTML report (plots: joints, tilt, cmd vs measured vel, effort, falls) | cloud | WP2-01 | done (`bipedlab report`, CI artifact) | single file, opens offline; generated in CI as artifact |
| WP2-03 | Control-timing analysis (sim time vs wall time, jitter histogram) | cloud | WP2-01 | later | report shows p50/p99 control period |
| WP2-04 | Turkish guide to reading reports | cloud | WP2-02 | done (`docs/learning/02-telemetri-raporu-okuma.md`; owner sign-off pending) | owner sign-off |
| MAC-05 | Live viewer overlay of telemetry on M5 | mac-m5 | WP2-01 | later | owner sees live plots |

## Phase 3

| ID | Task | Env | Deps | Status | Acceptance criteria |
|---|---|---|---|---|---|
| WP3-01 | Choose PPO library (SB3 vs rsl-rl) by a 30-min CPU benchmark | cloud | WP1-06 | later | ADR with measured throughput |
| WP3-02 | gymnasium env reproducing 61/14 contract + reward terms | cloud | WP3-01 | later | official policy inside env meets G1.3 |
| WP3-03 | PPO smoke train job (nightly, ≤ 5 min) | cloud | WP3-02 | later | mean reward rises over 3 seeds |
| WP3-04 | Full training run on M5 (CPU, then MPS comparison) | mac-m5 | WP3-03 | later | G3.3 |
| WP3-05 | DR ablation, ONNX export, comparison with official | mac-m5 | WP3-04 | later | G3.4, G3.5 |

## Phase 4–6 (outline; detailed at Phase 3 exit)

| ID | Task | Env | Status |
|---|---|---|---|
| WP4-01 | Requirements & sizing calculations (mass, torque, power, battery) | cloud | later |
| WP4-02 | Parametric MJCF generator for the custom biped (evolve `toy_biped_v0`) | cloud | later |
| WP4-03 | Actuator survey + identification plan (BAM-style) | cloud | later |
| WP4-04 | Rust runtime skeleton + sim bridge + safety supervisor model | cloud | later |
| WP4-05 | BOM draft (no purchase) | cloud | later → awaiting-owner |
| WP5-xx | Hardware tasks (all `hw`, owner approval) | hw | later |
| WP6-xx | Autonomy | – | later |
