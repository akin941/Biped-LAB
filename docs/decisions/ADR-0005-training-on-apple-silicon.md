# ADR-0005: RL training path — CPU MuJoCo + PPO first, no CUDA assumption

Status: accepted (2026-10-26)

## Context
- microduck_rl training requires CUDA (README; mjlab selects GPUs and fails without
  CUDA). It will not run unchanged on Apple Silicon.
- microduck-lab trains Microduck on CPU MuJoCo with Stable-Baselines3 PPO on Macs
  (code verified; performance claims unverified).
- No paid GPUs (owner constraint). M5: 32 GB unified memory, many CPU cores.

## Decision
- Phase 3 training uses CPU MuJoCo vectorized envs (process-parallel) + a PPO
  implementation running on CPU (torch CPU; MPS evaluated as an experiment, not assumed).
- Start with a known library (SB3 or rsl-rl) pinned, rather than writing PPO from
  scratch; write a small reference PPO only for learning material.
- The env reproduces the upstream 61/14 contract so policies are comparable with
  the official ones.
- Long runs happen on the Mac (or the self-hosted CPU runner, ≤ 8 GB RAM), never on
  every CI commit.

## Consequences
Training is slower than 4096 GPU envs; experiments are scaled down (fewer envs,
curricula). Results are compared against official pretrained policies in the same
evaluation harness.
