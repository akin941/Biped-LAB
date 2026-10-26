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
zero command. The zero-action baseline falls in every env.

### Per-term reward profile (weight × value, per second; gha-linux, second CI run)
For scale, the positive terms max out at 2.0 each (pose 1.0; air_time 6.0 for both
feet).

| Term | BAM, vx 0.4, noise | BAM, cmd 0, noise | Zero action, BAM, vx 0.4 |
|---|---|---|---|
| track_linear_velocity | 0.995 | 1.997 | 0.642 |
| track_angular_velocity | 0.848 | 1.997 | 1.058 |
| upright | 1.891 | 1.994 | 0.953 |
| pose | 0.722 | 0.993 | 0.959 |
| head_pose_tracking | 1.955 | 1.965 | 1.994 |
| air_time | 1.279 | 0 | 0 |
| action_rate_l2 | −0.167 | −0.005 | 0 |
| head_pose_bias | −0.149 | −0.123 | −0.016 |
| body_ang_vel | −0.030 | 0.000 | −0.161 |
| others | ≥ −0.006 | 0 | ≥ −0.004 |

The MJCF rows are within a few percent of the BAM rows (CI log of PR #18).

How to read it:
- **At a zero command** every positive term is within 1 % of its maximum and every
  penalty is near zero. This is what a trained policy scores in its training env.
- **While walking at 0.4 m/s**, upright and head tracking stay near their maxima,
  and air_time shows a real gait (about 0.4 feet per step inside the 125–300 ms
  window).
- **Linear tracking is half its maximum while walking**, and this is not near the
  maximum. It is what a 0.17–0.18 m/s walk against a 0.4 command must score:
  exp(−0.22²/0.1) ≈ 0.62 from the vx error alone, with lateral and vertical sway
  lowering it further. It reflects the policy's own under-tracking, which upstream's
  sim shows too (ADR-0011). It is not a defect of the env.

## Verdict
- **WP3-02 acceptance criterion: met.** The official policy inside the env
  reproduces upstream's number (0.4 → ~0.17 m/s, no falls).
- **The spec's extra profile check is met at a zero command, but not for linear
  tracking while walking.** That shortfall is explained above and left as written,
  not relaxed. Our own policy's bar remains G1.3 (ADR-0011).
