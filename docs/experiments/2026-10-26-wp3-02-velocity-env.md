# WP3-02: upstream velocity task as a CPU env, and G3.1 check

**Environment:** gha-linux (ubuntu-24.04 runner, PR #18 CI, job `pretrained-policy`)
unless marked cloud. Not run on the Mac.

## What was built
`bipedlab.rl.velocity_env.MicroduckVelocityEnv` implements the **v1** rows of
`docs/research/velocity-task-spec.md`: the flat velocity task that `alpha_walking`
was trained on. Throughput in the cloud container, 8 envs, serial: ~2600 env steps/s
with MJCF actuators and ~1100 with BAM.

## G3.1: official `alpha_walking` inside our env
`scripts/eval_velocity_env.py alpha_walking.onnx --envs 8`. Commands are frozen;
the episode is 10 s and the speed is the body-frame vx averaged after 2 s. The
curriculum weights are the end-of-curriculum values.

| Policy | Actuator | Command vx | Obs noise | Falls / 8 | Mean vx (m/s) | Reward / step |
|---|---|---|---|---|---|---|
| alpha_walking | BAM | 0.4 | on | 0 | 0.183 | 0.147 |
| alpha_walking | BAM | 0.4 | off | 0 | 0.168 | 0.132 |
| alpha_walking | BAM | 0 | on | 0 | 0.000 | 0.176 |
| alpha_walking | BAM | 0 | off | 0 | 0.000 | 0.177 |
| zero action | BAM | 0.4 | on | 8 | — | 0.109 |
| alpha_walking | MJCF | 0.4 | on | 0 | 0.193 | 0.148 |
| alpha_walking | MJCF | 0.4 | off | 0 | 0.184 | 0.136 |
| alpha_walking | MJCF | 0 | on | 0 | 0.000 | 0.175 |
| alpha_walking | MJCF | 0 | off | 0 | 0.000 | 0.176 |
| zero action | MJCF | 0.4 | on | 8 | — | 0.099 |

The speed part of the criterion is **met**. With BAM and a 0.4 m/s command, our env
gives 0.17–0.18 m/s with no falls. Upstream's own sim evaluation gives ~0.17 m/s, and
our earlier rollout harness gave 0.155 m/s (ADR-0011). The policy holds still at a
zero command. The zero-action baseline falls in every env. The per-term reward
profile is still to be recorded from the next CI run on this PR; see below.
