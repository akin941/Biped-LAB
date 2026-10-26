# ADR-0002: Python 3.12 + uv + CPU MuJoCo 3.10 as the simulation baseline

Status: accepted (2026-10-26)

## Context
- Official Microduck training (microduck_rl) uses mjlab 1.3.0 on MuJoCo-Warp and
  requires CUDA. Its inference script and microduck-lab use plain CPU MuJoCo.
- microduck_rl pins Python `>=3.12,<3.13` (BAM requires < 3.13) and resolves
  mujoco 3.10.0, numpy 2.4.1, onnxruntime 1.24.4.
- All three have macOS arm64 wheels (verified in our `uv.lock`), Linux x86_64 and aarch64.

## Decision
- Python 3.12, `uv` with committed `uv.lock`, exact version pins.
- MuJoCo **3.10.0** CPU (same version upstream trains against) for all our sims.
- onnxruntime CPU execution provider for policy inference.
- No mjlab / MuJoCo-Warp / torch in the base install.

## Alternatives
- MJX (JAX) or MuJoCo-Warp: GPU-oriented; Warp needs CUDA, JAX-Metal is experimental. Rejected for baseline.
- Isaac Lab / Genesis: NVIDIA-centric or less aligned with upstream. Rejected.

## Consequences
Same code runs in cloud, CI (Linux + macOS runner) and on the M5. Upstream version
bumps are deliberate and tested. GPU-parallel training is out of scope until a new ADR.
