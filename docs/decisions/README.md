# Architecture decision records

Append-only. To change a decision, add a new ADR that supersedes the old one and
update the old one's status line only.

| ADR | Title | Status |
|---|---|---|
| [0001](ADR-0001-record-decisions.md) | Record decisions as ADRs in the repo | accepted |
| [0002](ADR-0002-python-uv-cpu-mujoco.md) | Python 3.12 + uv + CPU MuJoCo 3.10 as the simulation baseline | accepted |
| [0003](ADR-0003-upstream-as-pinned-data.md) | Upstream projects are pinned, untrusted data | accepted |
| [0004](ADR-0004-licensing.md) | Apache-2.0 for this repo; no third-party meshes; original robot design | accepted |
| [0005](ADR-0005-training-on-apple-silicon.md) | RL training path: CPU MuJoCo + PPO first, no CUDA assumption | accepted |
| [0006](ADR-0006-layered-safety.md) | Layered control; AI never commands motors directly | accepted |
| [0007](ADR-0007-rust-scope.md) | Rust only for the on-robot runtime, adopted in Phase 4 | accepted |
| [0008](ADR-0008-evidence-environments.md) | Every result is labelled with the environment that produced it | accepted |
| [0009](ADR-0009-no-ros2-docker-yet.md) | No ROS 2 now; Docker only for reproducible Linux CI if needed | accepted |
| [0010](ADR-0010-bam-actuator-port.md) | BAM XL330 actuator as a reviewed in-repo port, not an installed package | accepted |
| [0011](ADR-0011-g13-and-phase3-entry.md) | G1.3 stays as written; Phase 3 starts on simulator-fidelity evidence | accepted |
