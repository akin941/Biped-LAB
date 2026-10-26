# PPO library CPU benchmark: rsl-rl-lib vs Stable-Baselines3 (WP3-01)

```
env: cloud (4 vCPU Intel Xeon @ 2.1 GHz, Linux x86_64), revision baed27d + this PR
python 3.12.3, mujoco 3.10.0, numpy 2.5.4, torch 2.9.1 (PyPI wheel, CPU, 4 threads)
rsl-rl-lib 5.0.1, tensordict 0.10.0, stable-baselines3 2.7.1, gymnasium 1.2.3
command: python scripts/bench_ppo.py --envs 16 64 256 --iters 20   (run twice)
```

## Setup
- Environment: `bipedlab.rl.batch_env.MicroduckBatchEnv`, official `scene.xml`, MJCF
  actuators, 61-D observation and 14-D action of the upstream contract, 50 Hz control
  (4 × 5 ms physics steps), all envs stepped in one process. Placeholder stand task
  (alive + upright − action rate, fall ends the episode); WP3-02 brings upstream's
  rewards. Both libraries use the identical env through thin adapters.
- PPO settings: upstream's `MicroduckRlCfg` (actor and critic 512-256-128 ELU, 24
  steps per env, 5 epochs, 4 mini-batches, lr 1e-3, gamma 0.99, lambda 0.95, clip 0.2,
  entropy 0.01, value coefficient 1.0, grad-norm 1.0). rsl-rl also runs observation
  normalisation and the adaptive-KL lr schedule as upstream does; SB3 has neither
  built in and runs a fixed lr. SB3 additionally needs finite action bounds (±10 rad,
  clipped); rsl-rl passes actions raw.
- One warm-up iteration, then 20 timed iterations. "env" is time inside `env.step`;
  the rest is action sampling, storage and the PPO update.

## Results

| run | library | envs | samples/s | env s | non-env s | env share |
|---|---|---|---|---|---|---|
| 1 | rsl-rl | 16 | 1238 | 3.0 | 3.2 | 48 % |
| 1 | SB3 | 16 | 1283 | 3.2 | 2.8 | 53 % |
| 1 | rsl-rl | 64 | 1766 | 12.5 | 4.9 | 72 % |
| 1 | SB3 | 64 | 1877 | 12.1 | 4.2 | 74 % |
| 1 | rsl-rl | 256 | 1939 | 53.3 | 10.1 | 84 % |
| 1 | SB3 | 256 | 1966 | 52.9 | 9.6 | 85 % |
| 2 | rsl-rl | 16 | 1220 | 3.1 | 3.2 | 49 % |
| 2 | SB3 | 16 | 1253 | 3.2 | 2.9 | 53 % |
| 2 | rsl-rl | 64 | 1676 | 13.2 | 5.2 | 72 % |
| 2 | SB3 | 64 | 1818 | 12.5 | 4.4 | 74 % |
| 2 | rsl-rl | 256 | 1879 | 55.0 | 10.3 | 84 % |
| 2 | SB3 | 256 | 1833 | 56.4 | 10.7 | 84 % |

Raw env stepping (random actions, no learning): run 1 2823 / 2883 / 2412 steps/s and
run 2 2470 / 2394 / 2233 steps/s for 16 / 64 / 256 envs. MuJoCo alone on this model:
16.6k physics steps/s (`bipedlab.bench.physics_steps_per_s`), i.e. 4.1k control
steps/s, so ~35 % of env time is Python (observation building, reward, reset).

Mean step reward over the 20 iterations was identical between the two runs for each
library (same seeds), and rose with batch size for both, so both adapters learn.
This is a throughput check, not a learning comparison.

## Reading
- At small batches SB3 is ~3–8 % faster (lighter update: no observation normaliser,
  no KL schedule). At 256 envs the two are equal within run-to-run spread (±3 %).
- The environment is the bottleneck: 84 % of wall time at 256 envs. Parallelising
  env stepping (WP3-06) matters far more than the library choice.
- The choice therefore rests on fidelity to upstream (ADR-0012): rsl-rl-lib 5.0.1.
- These are cloud numbers. Apple Silicon numbers come from MAC-07.
