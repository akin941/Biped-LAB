# WP4-06: velocity env and PPO smoke run for our own biped_v1 (G4.3 preparation)

`bipedlab.rl.biped_velocity_env.BipedVelocityEnv` is the flat velocity task for the
generated `simulation/custom-biped/biped_v1.xml` (10 position-controlled joints,
XL330-class placeholder servos, 0.52 N·m force limit). It follows
`MicroduckVelocityEnv` (timing, resets, twist commands, observation noise and lag,
asymmetric critic, reward family, optional DR) without Microduck's head joints and
head/body-pose commands. Contract: actor 39 → 10 actions, critic 48.

Differences from the Microduck task, chosen for this robot (cloud, not tuned yet):
- action = stand pose + 0.5 × policy output (Microduck uses 1.0);
- forward commands up to 0.3 m/s, sideways 0.2 m/s, turning 1.0 rad/s;
- air-time reward counts single support only; feet touching each other is penalised;
- fall = tilt over 60° or trunk below half the stand height;
- DR: foot friction, trunk mass and CoM, armature, encoder bias, IMU tilt, pushes
  (0.2 m/s; Microduck's head-CoM and BAM terms do not exist here).

## Tests (cloud container, Linux x86_64, 4 CPUs)

`uv run pytest tests/unit/test_biped_velocity_env.py`: 10 passed. They cover the
contract, keyframe resets with random yaw, standing 3 s with zero actions, fall and
auto-reset, seed determinism, shard and `ParallelBatchEnv` equality with the serial
env, command resampling and freezing, the tracking reward and DR ranges.

## PPO smoke run (cloud container)

`scripts/train_ppo.py --robot biped_v1 --envs 64 --iters 60 --seed N --threads 1`,
rsl-rl-lib 5.0.1, torch 2.9.1 (PyPI wheel; the PyTorch CPU index is blocked in the
container). Criterion as WP3-03: the mean reward of the last 15 iterations is above
the first 15.

| Seed | reward/step first → last quarter | episode length (steps) first → last it. | wall |
|---|---|---|---|
| 0 | −0.0155 → −0.0062 | 22 → 43 | 54 s |
| 1 | −0.0137 → −0.0049 | 21 → 44 | 54 s |
| 2 | −0.0152 → −0.0054 | 20 → 39 | 54 s |
| 0, `--dr` | −0.0171 → −0.0051 | 22 → 42 | 52 s |

**Criterion met on all 3 seeds and with DR (cloud).** The same runs are in the
`nightly-train` workflow (`biped-ppo-smoke` job) for gha-linux evidence. As with
WP3-03, the rise comes mostly from the action-rate penalty falling as the initial
std of 1.0 narrows; episodes get longer, the robot still falls within ~1 s. This is
a plumbing check, not a walking result. Walking is MAC-08.

A 10-iteration run with `--workers 2 --run-dir` saved checkpoints and exported a
39 → 10 ONNX policy, which `scripts/eval_biped.py` loaded and ran (finite rollouts).

## Knee torque, first look

During these early runs the knee reached its 0.52 N·m force limit every iteration
(random exploration actions), while standing with zero actions needs only about
0.04 N·m (p95, eval script). Whether a learned gait stays under the XL330 limit is
what MAC-08 measures (`knee_torque` in `eval_biped.py`); WP4-01 predicted 0.34 N·m
needed for walking (0.68 N·m with the 2× margin).
