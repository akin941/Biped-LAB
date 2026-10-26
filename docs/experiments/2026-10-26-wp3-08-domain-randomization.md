# WP3-08: domain randomization in the velocity env (cloud part of WP3-05)

**Environment:** cloud (container, 4 CPUs) unless marked gha-linux. Not run on the Mac.

## What was built
`MicroduckVelocityEnv(dr=True)` implements every **DR** row of
`docs/research/velocity-task-spec.md`, with the values upstream's velocity task uses
(`microduck_velocity_env_cfg.py`, flags that are on):

| Randomization | Range | When | Seen by |
|---|---|---|---|
| Velocity push (world x, y) | ±0.3 m/s added | every 3–6 s per env | physics |
| Foot friction (both feet, one value) | 0.7–1.3 | startup | physics |
| Trunk mass and inertia, together | ×0.95–1.05 | startup | physics |
| Trunk CoM | ±3 → ±15 mm by 1500 it | reset | physics |
| Head-assembly CoM (5 bodies) | ±3 → ±10 mm by 1000 it | reset | physics |
| BAM friction budget (not viscous) | ×0.9–1.1 per env | reset | physics (BAM only) |
| Armature, per hinge dof | ×0.9–1.1 | reset | physics |
| IMU mounting error, random axis | ≤ 6° | startup | actor gyro and gravity |
| Encoder bias, per joint | ±0.015 rad | startup | actor joint_pos |

`dr=False` (the default) leaves the env bit for bit as before: the existing tests are
unchanged and pass. DR draws from separate per-env streams, so commands and reset
poses match `dr=False` for the same seed. The approximations are listed in the
spec's Implementation section.

## Checks
- Tests (`tests/smoke/test_velocity_env.py`, 5 new): ranges and per-env writes into
  the model, CoM curriculum, actor sees exactly R·gyro, R·gravity and joint_pos + bias
  while the critic stays clean, push bounds and timing, determinism across shards and
  worker processes, and that DR changes the physics.
- PPO smoke with DR (cloud, 64 envs, 60 it, 2 workers): seed 0 −0.0066 → 0.0072,
  seed 1 −0.0076 → 0.0058, both rose (~50 s each). Nightly now runs seed 0 with DR.
- Throughput (cloud, 8 envs, serial): MJCF 2807 → 2453 steps/s with DR (−13 %),
  BAM 1157 → 1143 (−1 %).
- `alpha_walking` under DR: run in CI (gha-linux) because the weights cannot be
  fetched in the cloud container. Results below.

## alpha_walking under DR (gha-linux)
Pending the CI run of this PR.
