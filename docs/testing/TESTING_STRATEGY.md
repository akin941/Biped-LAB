# CI and testing strategy

Every result carries an **environment label**. A result only counts as evidence
for the environment that produced it.

| Label | Where | Counts as |
|---|---|---|
| `cloud` | Claude Code cloud container (Linux x86_64, CPU only) | Linux CPU evidence |
| `gha-linux` | GitHub-hosted `ubuntu-24.04` | Linux CPU evidence |
| `gha-macos` | GitHub-hosted `macos-15` (Apple Silicon, not M5, no display) | macOS arm64 *install/CPU* evidence only |
| `mac-m5` | Owner's MacBook Pro M5, local session | the only source of Mac viewer / Metal / MPS / M5 performance evidence |
| `self-hosted` | Owner's x86_64 Ubuntu runner (~8 GB RAM) | not used yet; reserved for long CPU jobs, never GPU training |
| `hw` | Physical robot | the only source of hardware evidence (Phase 5+) |

## Test tiers

| Tier | Marker / job | Budget | Runs on | Trigger |
|---|---|---|---|---|
| Static | `ci/static`: ruff, ruff format, mypy --strict | < 1 min | gha-linux | every push/PR |
| Unit | `pytest -m "not upstream and not slow"` | < 30 s | gha-linux, gha-macos | every push/PR |
| CPU sim smoke | `pytest -m upstream` (pinned Microduck MJCF) | < 1 min | gha-linux, gha-macos | every push/PR |
| Integration | `slow` marker (policy rollouts, multi-second sims) | < 10 min | gha-linux | PR + nightly once added |
| Native macOS | `docs/handoffs/MAC_LOCAL_HANDOFF.md` procedures | manual | mac-m5 | when a handoff task is pending |
| RL training | `learning/training/` configs, never in default CI | hours | mac-m5 / self-hosted (CPU) | manual, owner-visible |
| Hardware | HIL procedures (Phase 5) | manual | hw | owner approval |
| Security | gitleaks, pip-audit, dependency-review | < 2 min | gha-linux | every push/PR + weekly |

## Principles

- **Small deterministic fixtures.** `simulation/custom-biped/toy_biped_v0.xml` is our
  own model, so unit tests never depend on upstream availability.
- **Determinism is tested.** Same inputs → same state digest (`sha256` of qpos/qvel)
  on the same platform. Digests are *not* compared across platforms (float
  differences between x86_64 and arm64 are expected).
- **Characterization tests** pin upstream facts we depend on (actuator names, gains,
  mass, keyframes). An upstream bump that changes them must fail loudly.
- **No training in CI.** At most a few hundred PPO steps as a "does it run" check,
  marked `slow`.
- **Performance** numbers are recorded with their environment label and never
  compared across labels. Cloud steps/s do not predict M5 steps/s.
- **Never** skip, xfail, or loosen a test to get green; fix it or record a blocker.

## Evidence record format

Each completed task links an evidence entry (in `docs/experiments/` or the PR):

```
task: WP1-03
revision: <git sha>
env: cloud | gha-linux | gha-macos | mac-m5 | hw
command: uv run pytest -m upstream
result: 4 passed
date: YYYY-MM-DD
notes: ...
```
