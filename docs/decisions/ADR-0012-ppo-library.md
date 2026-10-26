# ADR-0012: PPO library is rsl-rl-lib 5.0.1, on CPU torch

Status: accepted (2026-10-26)

## Context
ADR-0005 left the PPO library open between Stable-Baselines3 and rsl-rl, to be chosen
by a CPU benchmark (WP3-01). Facts found while preparing it:
- Upstream `microduck_rl` trains every Microduck policy with **rsl-rl-lib 5.0.1** and
  torch 2.9.1 (its `uv.lock`), with one PPO configuration across tasks: actor and
  critic 512-256-128 ELU with observation normalisation, 24 steps per env, 5 epochs,
  4 mini-batches, lr 1e-3 with the adaptive-KL schedule (target 0.01), gamma 0.99,
  lambda 0.95, clip 0.2, entropy 0.01, scalar Gaussian std starting at 1.0, and
  left/right symmetry augmentation for walking.
- microduck-lab (community) trains with SB3 on Macs (ADR-0005).

Benchmark (`scripts/bench_ppo.py`, environment `cloud`: 4 vCPU Intel Xeon 2.1 GHz,
torch 2.9.1 with 4 threads, two runs; details in
`docs/experiments/2026-10-26-wp3-01-ppo-library-benchmark.md`). Both libraries train
the same `MicroduckBatchEnv` with upstream's settings above (SB3 has no adaptive-KL
schedule or built-in observation normalisation, so it runs a fixed lr without
normalisation, which only makes its update cheaper):

| envs | rsl-rl samples/s | SB3 samples/s | env share of wall time |
|---|---|---|---|
| 16 | 1220–1238 | 1253–1283 | 47–53 % |
| 64 | 1676–1766 | 1818–1877 | 72–74 % |
| 256 | 1879–1939 | 1833–1966 | 84 % |

Raw env stepping without PPO: 2.2–2.9k steps/s (one process); MuJoCo alone does
16.6k physics steps/s = 4.1k control steps/s, so ~35 % of env time is Python.

## Decision
- Use **rsl-rl-lib 5.0.1** with torch 2.9.1 (CPU), tensordict 0.10.0, pinned to
  upstream's lock. Throughput does not decide it (the two are within ±5 % at the batch
  sizes we will train with, run-to-run spread is about as large); fidelity does:
  upstream's trainer and configuration can be reused unchanged, so G3.1 compares
  like with like, and symmetry augmentation, observation normalisation, the KL
  schedule and the ONNX export path come with it.
- Our environment implements rsl-rl's `VecEnv` on top of `bipedlab.rl.batch_env`
  (NumPy, no torch import), so the env stays library-agnostic and testable in CI
  without torch.
- rsl-rl runs with `log_dir=None` or a local TensorBoard directory only; its wandb and
  neptune writers are not used (no external services, no accounts).
- Throughput work goes into the environment, not the library: WP3-06 parallelises env
  stepping across processes.
- torch is added to the project as an optional `train` extra when the training env
  lands (WP3-02). On Linux CI it is installed from the PyTorch CPU index (the PyPI
  Linux wheel pulls ~7 GB of CUDA libraries); on macOS from PyPI.

## Consequences
- At ~2k samples/s, upstream's walking budget (4096 envs × 24 steps × up to 50k
  iterations ≈ 4.9e9 samples) is out of reach on one CPU process (~28 days). Our
  runs use fewer envs and iterations; G3.3 is judged on behaviour, not on matching
  upstream's sample count. M5 numbers come from MAC-07 and WP3-04, never from here.
- rsl-rl is a moving project (5.5.1 at time of writing). We stay on 5.0.1 to match
  upstream; bumping it is its own PR with a re-run of the benchmark and G3.1.
- SB3 remains a fallback if a later Mac measurement (MAC-07) shows it clearly faster.
