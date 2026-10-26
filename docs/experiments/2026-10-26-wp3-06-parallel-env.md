# Process-parallel batch env: scaling (WP3-06)

```
envs: cloud (4 vCPU Xeon 2.1 GHz); gha-linux and gha-macos from CI run 40319571072 (PR #16)
code: bipedlab.rl.parallel_env.ParallelBatchEnv, spawn workers, pipes
command: bipedlab bench microduck --env-workers 0 1 2 N --num-envs K --seconds 4
```

## Raw env steps/s (random actions)

| env | logical CPUs | envs | serial | 1 worker | 2 workers | 3 workers | 4 workers |
|---|---|---|---|---|---|---|---|
| cloud | 4 | 64 | 2213 | 2246 (1.02×) | 3766 (1.70×) | – | 5422 (2.45×) |
| cloud | 4 | 256 | 2298 | – | – | 5498 (2.39×) | 6023 (2.62×) |
| gha-linux | 4 | 128 | 6363 | 6317 (0.99×) | 11660 (1.83×) | – | 13247 (2.08×) |
| gha-linux, 2nd run (run 40320521618) | 4 | 128 | 2873 | 2716 (0.95×) | 5503 (1.92×) | – | 6596 (2.30×) |
| gha-macos (M1 VM) | 3 | 128 | 4524 | 4376 (0.97×) | 6135 (1.36×) | 7496 (1.66×) | – |

One worker costs ≤ 3 % against the serial env, so pipe and pickling overhead is
small. Two workers scale well on Linux (0.85–0.92 per worker); beyond that the
gain flattens.

## PPO end to end (cloud, rsl-rl 5.0.1, 256 envs, 10 iterations)

| env workers | samples/s | env s | update + rollout s |
|---|---|---|---|
| serial | 1777 | 28.9 | 5.6 |
| 3 | 3346 | 12.7 | 5.7 |
| 4 | 3422 | 11.7 | 6.3 |
| 4, torch 2 threads | 3021 | 11.8 | 8.5 |
| 4, torch 1 thread | 2774 | 10.3 | 11.8 |

PPO throughput roughly doubles (1.9×). The update keeps torch's default thread count:
fewer threads slow the update more than they free the workers.

## Against the acceptance criterion
WP3-06 asks for ≥ 0.7 × cores on gha-linux, i.e. ≥ 2.8× with 4 workers. Measured:
**2.08× — not met.** cloud 2.62× (0.65 per core) and gha-macos 1.66× on 3 cores (0.55)
fall short too. The second gha-linux run printed the topology: AMD EPYC 7763, **2 physical cores ×
2 threads**. Per physical core the env scales well (2 workers: 1.83–1.92×), and the
SMT siblings add only 0.15–0.4×. The runner's absolute serial rate also varies
2.2× between runs (6363 vs 2873 steps/s), so single CI numbers are noisy. The
criterion is left as written and recorded as not met; whether "cores" should mean
physical cores is a question for the M5 measurement (MAC-07), which has no SMT. Equivalence with the serial
env (the criterion's second half) is met and covered by
`tests/smoke/test_parallel_env.py`.

Next levers, in order: cut the ~35 % Python share of env time (vectorised
observation and reward over the batch) and keep the parallel env for the M5's
performance cores (MAC-07 measures it there).
