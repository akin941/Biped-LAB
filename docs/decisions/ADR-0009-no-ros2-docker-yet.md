# ADR-0009: No ROS 2 now; Docker only for reproducible Linux CI if needed

Status: accepted (2026-10-26)

## Context
Microduck itself does not use ROS 2 (custom JSON-RPC daemons). Our needs through
Phase 4 are one robot, one process graph, 50 Hz control. ROS 2 on macOS is poorly
supported; Docker on Mac cannot use Metal and adds a VM.

## Decision
- No ROS 2. Revisit in Phase 6 if navigation/perception stacks (Nav2, SLAM) are needed.
- No Docker for the main workflow; `uv` already gives reproducible envs. A Dockerfile
  may be added for the self-hosted Linux runner if its environment drifts.

## Consequences
Lighter stack; we lose ROS tooling (rviz, bags) and replace it with our own telemetry
(Phase 2).
