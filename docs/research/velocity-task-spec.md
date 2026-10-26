# Upstream Microduck velocity task: specification for our CPU env (WP3-02)

Source, read as data (not executed):
- `microduck_rl` @ `b02fdea` — `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py`
  (`make_microduck_velocity_env_cfg`, flat: `rough=False`) and `tasks/mdp.py`.
- `mjlab` 1.3.0 wheel from PyPI, sha256 `e3b544bf…fe47cf` (the version in upstream's
  `uv.lock`) — `tasks/velocity/velocity_env_cfg.py`, `tasks/velocity/mdp/*.py`,
  `envs/mdp/*.py`, `managers/reward_manager.py`.

This is what `alpha_walking`-style policies were trained on. Each row says whether our
env reproduces it in the first version (**v1**), later (**DR** = with the domain-
randomization work, WP3-05), or not at all, and why.

## Timing, model, actions
| Item | Upstream | Ours |
|---|---|---|
| Physics | MuJoCo-Warp, dt 5 ms, decimation 4 → 50 Hz; solver iterations 10, ls 20 (MJCF has 100 / 50) | CPU MuJoCo, same dt, decimation and solver iterations — **v1** (Warp vs CPU solver differences remain) |
| Model | `scene_walk.xml` (walk model; sensors `imu_ang_vel`, `imu_lin_vel`, `root_angmom`; sites `left_foot`, `right_foot`), plane terrain | same file — **v1** |
| Actuator | BAM XL330 M6 with friction DR hook (`FrictionDRBamActuatorCfg`) | our BAM port (ADR-0010) — **v1** option `--actuator bam`; MJCF position default for speed |
| Action | 14 joint targets, `target = HOME + 1.0 × action`, no clipping | same — **v1** |
| Episode | 20 s | 20 s — **v1** |
| Reset | root z ∈ [0.12, 0.13] m, x/y ∈ ±0.5 m, yaw ∈ ±π, joints at HOME, zero velocity | same — **v1** |

## Observations (actor, 61 = contract)
| Term | Noise (uniform ±) | Delay | Ours |
|---|---|---|---|
| base_ang_vel (IMU gyro) | 0.03 | 0–1 step, resampled every 64 steps | **v1** noise + delay |
| projected_gravity | 0.01 | 0–1 step, every 64 steps | **v1** |
| joint_pos − HOME | 0.001 | — ; + per-env encoder bias ±0.015 rad | **v1** noise; bias **DR** |
| joint_vel | 0.25 | exactly 1 step | **v1** |
| last_action | — | — | **v1** |
| twist (3), head_pose (4), body_pose (6) commands | — | — | **v1** |
| IMU mounting error (random axis ≤ 6°) on gyro and gravity | — | per env | **DR** |

Critic (privileged, uncorrupted): actor terms without noise + base_lin_vel, foot height,
foot air time, foot contact, foot contact forces. Ours: **v1** with base_lin_vel, foot
contact and air time; foot height from site z (flat ground, no ray sensor).

## Commands
- **twist** (resample every 3–8 s): vx ∈ ±0.4, vy ∈ ±0.3, wz ∈ ±1.0 m/s, rad/s,
  independent uniform. Then: 20 % "forward" envs → vx = max(|vx|, 0.3), vy = wz = 0;
  15 % "turn in place" → vx = vy = 0, |wz| ∈ [0.4, 1.0] random sign; standing envs
  (2 % → 25 % by curriculum) → all zero. No heading envs. — **v1**
- **head_pose** (resample 2–5 s): deltas from HOME for neck_pitch, head_pitch,
  head_yaw, head_roll; ranges widen by curriculum from ±(0.05, 0.05, 0.07, 0.015) to
  ±(1.10, 1.10, 1.40, 0.31). — **v1**
- **body_pose** (resample 2–5 s): ±0.005 m (x, y, z), ±0.05 rad (roll, pitch, yaw);
  weight 0 in this task, kept so the obs slot is alive. — **v1**

## Rewards (each term × weight × dt, dt = 0.02 s)
| Term | Weight | Definition | Ours |
|---|---|---|---|
| track_linear_velocity | 2.0 | exp(−(‖cmd_xy − v_xy‖² + v_z²) / 0.1), body frame | **v1** |
| track_angular_velocity | 2.0 | exp(−((cmd_wz − ω_z)² + ‖ω_xy‖²) / 0.5) | **v1** |
| upright (trunk_base) | 2.0 | exp(−‖g_xy‖² / 0.05), g = projected gravity | **v1** |
| pose (legs only) | 1.0 | exp(−mean((q − HOME)² / σ²)); σ standing (cmd < 0.01): hip_yaw 0.1, hip_roll 0.05, hip_pitch 0.15, knee 0.15, ankle 0.1; walking: 0.3, 0.05, 0.4, 0.4, 0.25 | **v1** |
| body_ang_vel (trunk_base) | −0.05 | ‖ω_xy world‖² | **v1** |
| angular_momentum | −0.02 | ‖L‖², L = `root_angmom` (subtreeangmom of trunk_base) | **v1** |
| dof_pos_limits | −1.0 | Σ distance outside soft limits (90 % of range, centred) | **v1** |
| action_rate_l2 | −0.1 → −1.0 | ‖a − a_prev‖²; curriculum −0.1 (0), −0.2 (500 it), −0.4 (750), −0.6 (1000), −0.8 (1250), −1.0 (1500) | **v1** |
| air_time | 3.0 | Σ_feet [0.125 < current air time < 0.300 s], only if ‖cmd_xy‖ + |cmd_wz| > 0.01 | **v1** |
| foot_clearance | −2.0 | Σ_feet |h − 0.02| · ‖v_foot,xy‖, gated by command > 0.01 | **v1** (h = foot site height on flat ground) |
| foot_swing_height | −0.25 | at each touchdown: (peak h / 0.02 − 1)², gated | **v1** |
| foot_slip | −0.1 | Σ_feet in contact ‖v_foot,xy‖², gated | **v1** |
| self_collisions | −1.0 | number of trunk-subtree self contacts | **v1** |
| head_pose_tracking | 2.0 | mean_j exp(−((q_j − HOME_j − cmd_j)/0.5)²) over the 4 head joints | **v1** |
| head_pose_bias | 0 → 3.0 | −mean |EMA_1s(err)|; curriculum 0 until 600 it, 1.0, 2.0 (1000), 3.0 (1500) | **v1** |
| body_pose_tracking | 0 | — | omitted (weight 0) |

"it" = PPO iterations of 24 env steps; upstream counts curriculum steps as env steps
(`it × 24`), so our schedule is per env step too.

## Terminations
- time_out at 20 s (time-out, bootstrapped) — **v1**
- fell_over: tilt > 70° — **v1**
- nan_state: non-finite robot state — **v1**

## Events and domain randomization (all **DR**, WP3-05)
Velocity pushes every 3–6 s (Δv_xy ∈ ±0.3 m/s); foot friction ∈ [0.7, 1.3] (startup);
encoder bias ±0.015 rad (startup); trunk CoM ±3 → ±15 mm and head-assembly CoM
±3 → ±10 mm (reset, curriculum); trunk mass+inertia ×[0.95, 1.05] (startup);
BAM friction ×[0.9, 1.1] (reset); armature ×[0.9, 1.1] (reset); IMU mounting error.

## Curricula
action_rate weight, standing-env fraction (0.02 → 0.25 by 2000 it), head-pose ranges
(5 stages to 2000 it), head_pose_bias weight, CoM ranges. All **v1** except the CoM
ones (**DR**).

## PPO
Same as ADR-0012 (512-256-128 ELU, obs normalisation, 24 steps, 5 epochs, 4 mini-
batches, lr 1e-3 adaptive KL 0.01, γ 0.99, λ 0.95, clip 0.2, entropy 0.01, std 1.0).
Upstream: 4096 envs, up to 50 000 iterations, symmetry augmentation off for this task.

## Validation (G3.1, ADR-0011)
Inside our env, the official `alpha_walking` policy must reproduce upstream's sim
number (vx command 0.4 → ~0.17 m/s, no falls) and score a per-term reward profile
consistent with a trained policy (tracking and upright terms near their maxima).

## Implementation (`bipedlab.rl.velocity_env`)
`MicroduckVelocityEnv` implements every **v1** row above. Known approximations:
- Contact, air time and swing peaks are sampled once per control step (20 ms); mjlab's
  contact sensor updates every 5 ms substep. Air-time windows (125–300 ms) are wide
  relative to 20 ms; a touchdown shorter than one control step is missed.
- self_collisions counts robot-robot contacts at the control step; upstream counts
  substeps where a self-contact force exceeds 10 N.
- CPU MuJoCo instead of MuJoCo-Warp (same model, timestep, solver iterations).
- The actor's joint_vel delay is applied only with noise on; `noise=False` gives the
  clean critic terms (used by tests and diagnostics).

Domain randomization (`dr=True`, off by default) implements every **DR** row with
upstream's ranges, sampling modes and curricula. Its draws come from separate per-env
streams, so commands and reset poses are the same as with `dr=False`. Additional
approximations:
- All envs share one `MjModel` (a copy is ~85 MB); each env's mass, CoM, armature and
  foot friction are written into it before that env is reset or stepped. MuJoCo's
  derived constants (`mj_setConst`) are not recomputed after these writes.
- Pushes are applied at the end of a control step, after observations, so they act
  from the next physics step; mjlab applies them just before observing.
- Foot friction sets axis 0 of both foot geoms. MuJoCo takes the larger of the two
  geoms' friction, and our floor has 1.0, so the effective range is [1.0, 1.3];
  upstream trains on mjlab's terrain plane, whose friction we have not checked.
- Upstream's base-template `base_com` event names no body for Microduck and is left
  out.

Throughput (cloud container, 8 envs, zero actions, serial): ~2600 env steps/s with
MJCF position actuators, ~1100 with BAM. `ParallelBatchEnv(env_cls=MicroduckVelocityEnv)`
shards it across processes with identical results.
