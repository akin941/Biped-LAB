# Batched observation and reward in the RL env (WP3-07)

```
envs: cloud (4 vCPU Xeon 2.1 GHz); gha-linux and gha-macos from PR #17 CI
  (runs 40321952751 = first commit, 40322974675 = with batched targets)
command: python scripts/ab_batch_env.py <main's batch_env.py> --num-envs 128 --rounds 3
```

## Change
Profiling the per-env loop (cloud, 64 envs) showed `np.cross` inside
`ObservationBuilder` at ~24 % of env time. `MicroduckBatchEnv` now builds the 61-D
observation and the reward once over the batch, sets all position targets with one
array operation and does the 4 physics steps with one `mj_step(nstep=4)` call.
`ObservationBuilder` itself is unchanged, so ONNX evaluations are unaffected.

Equivalence: every row is bitwise equal to `ObservationBuilder`
(`test_batched_observation_matches_observation_builder`), and an A/B against the
previous implementation over 60 steps with falls and time-limit resets gave identical
observations, dones and rewards (cloud).

## Speed (old and new alternated on the same machine)

| env | rounds (new / old) | median |
|---|---|---|
| cloud, first commit | 1.367, 1.305 | 1.336 |
| cloud, batched targets | 1.322, 1.372, 1.214 | 1.322 |
| gha-linux, first commit | 1.271, 1.268, 1.269 | **1.269** |
| gha-linux, batched targets | 1.282, 1.282, 1.230 | **1.282** |
| gha-macos, first commit | 1.274, 1.192, 1.341 | 1.274 |
| gha-macos, batched targets | –, 1.160, 1.252 | 1.160 |

After the change MuJoCo's `mj_step` is ~90 % of env time (cloud profile, 128 envs),
so little Python is left to remove in this process.

## Against the acceptance criterion
WP3-07 asks for +30 % serial env steps/s on gha-linux. Measured **+28 %** (+27 %
on the first commit): **not met**, criterion unchanged. The remaining gap is bounded
by physics time; further speed has to come from parallel workers (WP3-06) or from
cheaper physics settings, which would change the simulation and are out of scope here.
