"""Flat velocity task for our own biped, ``biped_v1`` (G4.3 preparation).

``BipedVelocityEnv`` is modelled on ``MicroduckVelocityEnv`` (WP3-02): the same
timing (50 Hz policy, 4 physics steps of 5 ms), random-yaw resets, twist commands
resampled every 3-8 s, actor observation noise with a one-step gyro/gravity lag,
asymmetric actor/critic observations and the same family of reward terms (velocity
tracking, upright, pose, air time, foot clearance/slip, action rate). It drops
everything that only exists on Microduck: the head joints and head/body-pose
commands, and the curricula tied to them.

The model is the generated MJCF of ``bipedlab.design.biped`` (10 position-controlled
joints, primitive geometry); by default the committed ``biped_v1.xml``. Joint order is
the actuator order: per side (left, right) hip_yaw, hip_roll, hip_pitch, knee,
ankle_pitch. Actions are offsets from the stand pose, scaled by ``ACTION_SCALE``.

Observation contract (actor, 39 floats):
    gyro (3) | projected gravity (3) | joint pos - stand pose (10) | joint vel (10) |
    last action (10) | command vx, vy, wz (3)
Critic (48): the clean actor terms, then base linear velocity in the body frame (3),
foot heights (2), feet air time (2), foot contacts (2).

Domain randomization (``dr=True``) uses the WP3-08 ranges where the parts exist on
biped_v1: foot friction, trunk mass and CoM, joint armature, encoder bias, IMU
mounting tilt and random pushes. All envs share one ``MjModel``; with DR each env's
parameters are written into it before that env is reset or stepped.

Target slew limit (WP4-11): the runtime's safety supervisor moves each joint target at
most ``TARGET_SLEW`` rad per period (10 rad/s, ~ the servo's no-load speed). The env
applies the same limit, so a policy trains on what the robot will execute, and the
``action_slew`` term penalises the part of each wanted step that the limit cuts off.
``slew_limited`` in each step result says which envs hit the limit, the metric the
runtime reports. ``target_slew=None`` turns the limit off (the WP4-06 behaviour).

Action delay (WP4-12, G4.3 latency robustness): ``action_delay=k`` applies each policy
action ``k`` control periods (``k`` x 20 ms) after it was produced; the observation's
"last action" stays the policy's own previous output, as on the robot. With DR, each
reset also draws an extra 0-``DR_ACTION_DELAY_MAX`` periods per env (WP4-16: the first
M5 policy halved its speed at 40 ms, MAC-08), from its own random stream so the other
DR draws are unchanged.

Each step also reports ``joint_torque``: the absolute actuator force per joint at the
end of the step, so a training or evaluation run can re-check the knee torque
requirement of WP4-01 with a learned gait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from bipedlab import REPO_ROOT
from bipedlab.design import biped
from bipedlab.rl.velocity_env import axis_angle_matrix, rotate, rotate_inverse, stage_value

DEFAULT_SCENE = REPO_ROOT / "simulation/custom-biped/biped_v1.xml"

PHYSICS_DT = 0.005
DECIMATION = 4
STEP_DT = PHYSICS_DT * DECIMATION  # 0.02 s, 50 Hz
EPISODE_S = 20.0
ACT_DIM = 2 * len(biped.LEG_JOINTS)  # 10
CMD_DIM = 3
OBS_DIM = 3 + 3 + 3 * ACT_DIM + CMD_DIM  # 39
CRITIC_DIM = OBS_DIM + 3 + 2 + 2 + 2  # 48
OBS_SLICES = {
    "gyro": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "joint_pos": slice(6, 6 + ACT_DIM),
    "joint_vel": slice(6 + ACT_DIM, 6 + 2 * ACT_DIM),
    "last_action": slice(6 + 2 * ACT_DIM, 6 + 3 * ACT_DIM),
    "command": slice(6 + 3 * ACT_DIM, OBS_DIM),
}
ACTION_SCALE = 0.5  # rad per unit action; Microduck uses 1.0 with stiffer servos
JOINT_INDEX = {f"{s}_{j}": k for k, (s, j) in
               enumerate((s, j) for s in biped.SIDES for j in biped.LEG_JOINTS)}  # fmt: skip
KNEES = np.array([JOINT_INDEX["left_knee"], JOINT_INDEX["right_knee"]])

# Reset: stand keyframe, random yaw and position, small joint noise.
RESET_XY = 0.5
RESET_JOINT_NOISE = 0.05  # rad
RESET_Z_LIFT = (0.0, 0.01)  # m above the keyframe height
FALL_TILT_DEG = 60.0
FALL_HEIGHT_FRACTION = 0.5  # trunk below this fraction of the stand height = fallen

# Actor observation corruption (WP3-02 values) and lag.
NOISE_HALF_WIDTH = np.repeat([0.03, 0.01, 0.001, 0.25], [3, 3, ACT_DIM, ACT_DIM])
LAG_RESAMPLE_STEPS = 64

# Commands, scaled from Microduck's ranges to a 0.13 m leg.
TWIST_RESAMPLE_S = (3.0, 8.0)
TWIST_RANGE = np.array([0.3, 0.2, 1.0])
FORWARD_FRACTION = 0.20
FORWARD_MIN_VX = 0.15
TURN_FRACTION = 0.15
TURN_WZ = (0.4, 1.0)
STANDING_FRACTION = 0.10
COMMAND_THRESHOLD = 0.01

# Runtime supervisor slew limit (runtime/src/safety.rs, max_target_speed 10 rad/s).
TARGET_SLEW = 10.0 * STEP_DT  # 0.2 rad per period

# Rewards. Weights follow MicroduckVelocityEnv except the action rate. Microduck ramps
# it from -0.1 to -0.6 over 1500 PPO iterations (stages keyed on env steps, 24 per
# iteration); on biped_v1 that let the deterministic policy jitter (WP4-11: after 300
# iterations 99.5 % of its periods hit the runtime slew limit), so it is -0.5 from the
# start. Stages stay a list so a later run can ramp again.
IT = 24
ACTION_RATE_STAGES = [(0, -0.5)]
WEIGHTS = {
    "track_linear_velocity": 2.0,
    "track_angular_velocity": 2.0,
    "upright": 2.0,
    "pose": 1.0,
    "body_ang_vel": -0.05,
    "dof_pos_limits": -1.0,
    "air_time": 3.0,
    "foot_clearance": -2.0,
    "foot_slip": -0.1,
    "feet_collision": -1.0,
    "action_slew": -1.0,  # (rad of wanted target step cut off by the slew limit)^2
}
TRACK_LIN_STD2 = 0.05
TRACK_ANG_STD2 = 0.5
UPRIGHT_STD2 = 0.05
POSE_STD_STANDING = np.tile([0.1, 0.05, 0.15, 0.15, 0.1], 2)
POSE_STD_WALKING = np.tile([0.3, 0.1, 0.4, 0.4, 0.3], 2)
SOFT_LIMIT_FACTOR = 0.9
AIR_TIME_WINDOW = (0.10, 0.30)
FOOT_TARGET_HEIGHT = 0.02

# Domain randomization (WP3-08 ranges).
PUSH_INTERVAL_S = (3.0, 6.0)
PUSH_DV = 0.2  # m/s; Microduck uses 0.3 at a similar size, ours is lighter-footed
FOOT_FRICTION = (0.7, 1.3)
ENCODER_BIAS = 0.015
MASS_SCALE = (0.95, 1.05)
TRUNK_COM = 0.01  # m, uniform per axis
ARMATURE_SCALE = (0.9, 1.1)
IMU_MAX_ANGLE_DEG = 6.0
DR_ACTION_DELAY_MAX = 2  # control periods (0-40 ms), drawn per reset


@dataclass
class BipedStepResult:
    obs: np.ndarray  # (N, 39) float32, after auto-reset
    critic_obs: np.ndarray  # (N, 48) float32, after auto-reset
    reward: np.ndarray  # (N,) float32
    terminated: np.ndarray  # (N,) bool: fell or non-finite
    truncated: np.ndarray  # (N,) bool: reached the time limit
    final_obs: np.ndarray  # (N, 39) before auto-reset
    terms: dict[str, np.ndarray]  # per-term rewards (N,), x weight x dt
    joint_torque: np.ndarray  # (N, 10) |actuator force| (N*m) at the end of the step
    slew_limited: np.ndarray  # (N,) bool: the slew limit cut a target step this step


@dataclass
class BipedVelocityEnv:
    scene: str | Path = DEFAULT_SCENE
    num_envs: int = 1
    seed: int = 0
    first_env: int = 0
    episode_s: float = EPISODE_S
    noise: bool = True
    common_step: int = 0
    dr: bool = False
    target_slew: float | None = TARGET_SLEW
    action_delay: int = 0
    # DR ablation knobs (WP4-18): extra DR delay range and foot friction range.
    dr_delay_max: int = DR_ACTION_DELAY_MAX
    dr_foot_friction: tuple[float, float] = FOOT_FRICTION
    model: mujoco.MjModel = field(init=False, repr=False)
    datas: list[mujoco.MjData] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.num_envs
        if n < 1:
            raise ValueError("num_envs must be at least 1")
        if self.action_delay < 0 or self.dr_delay_max < 0:
            raise ValueError("action_delay and dr_delay_max must be >= 0")
        m = mujoco.MjModel.from_xml_path(str(self.scene))
        if abs(m.opt.timestep - PHYSICS_DT) > 1e-12:
            raise ValueError(f"model timestep must be {PHYSICS_DT}")
        if m.nu != ACT_DIM:
            raise ValueError(f"model must have {ACT_DIM} actuators, has {m.nu}")
        self.model = m
        self._ids(m)
        self.datas = [mujoco.MjData(m) for _ in range(n)]
        self.max_episode_steps = round(self.episode_s / STEP_DT)
        self._rngs = [np.random.default_rng([self.seed, self.first_env + i]) for i in range(n)]
        if self.dr:
            self._init_dr()
        z = np.zeros
        self.episode_steps = z(n, dtype=np.int64)
        self.last_action = z((n, ACT_DIM), dtype=np.float32)
        self.last_target = np.tile(self.default_pose, (n, 1))
        max_delay = self.action_delay + (self.dr_delay_max if self.dr else 0)
        # history[k] = the action produced k periods ago (0 = this period).
        self.action_history = z((max_delay + 1, n, ACT_DIM), dtype=np.float32)
        self.delay = np.full(n, self.action_delay, dtype=np.int64)
        self._delay_rngs = [
            np.random.default_rng([self.seed, self.first_env + i, 2]) for i in range(n)
        ]
        self.twist = z((n, 3))
        self.twist_left = z(n)
        self._frozen: np.ndarray | None = None
        self.lag = z((n, 2), dtype=bool)
        self.lag_left = z(n, dtype=np.int64)
        self.prev_gyro = z((n, 3))
        self.prev_gravity = z((n, 3))
        self.prev_joint_vel = z((n, ACT_DIM))
        self.air_time = z((n, 2))
        self.contact = z((n, 2), dtype=bool)
        self._obs = z((n, OBS_DIM), dtype=np.float32)
        self._critic = z((n, CRITIC_DIM), dtype=np.float32)
        for i in range(n):
            self._reset_env(i)
        self._observe_into(np.arange(n))

    def _ids(self, m: mujoco.MjModel) -> None:
        def get(kind: mujoco.mjtObj, name: str) -> int:
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise ValueError(f"model lacks {name!r}")
            return int(i)

        B, G, S = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_SITE
        joints = m.actuator_trnid[:, 0]
        names = [m.joint(int(j)).name for j in joints]
        if names != list(JOINT_INDEX):
            raise ValueError(f"actuator order {names} differs from {list(JOINT_INDEX)}")
        self.qpos_idx = m.jnt_qposadr[joints].astype(int)
        self.qvel_idx = m.jnt_dofadr[joints].astype(int)
        lo, hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_LIMIT_FACTOR
        self.soft_lo, self.soft_hi = mid - half, mid + half
        key = get(mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.stand_qpos = m.key_qpos[key].copy()
        self.default_pose = self.stand_qpos[self.qpos_idx].copy()
        self.stand_height = float(self.stand_qpos[2])
        self.trunk_id = get(B, "trunk")
        self.site_ids = [get(S, f"{s}_foot") for s in biped.SIDES]
        self.foot_geoms = np.array([get(G, f"{s}_foot") for s in biped.SIDES])
        self.floor_geom = get(G, "floor")
        self.gyro_adr = int(m.sensor_adr[get(mujoco.mjtObj.mjOBJ_SENSOR, "gyro")])
        self.hinge_dofs = np.flatnonzero(m.jnt_type[m.dof_jntid] != mujoco.mjtJoint.mjJNT_FREE)

    # ----- domain randomization -------------------------------------------------

    def _init_dr(self) -> None:
        m, n = self.model, self.num_envs
        self._dr_rngs = [
            np.random.default_rng([self.seed, self.first_env + i, 1]) for i in range(n)
        ]
        t = self.trunk_id
        self._default_ipos = m.body_ipos[t].copy()
        self._default_mass = float(m.body_mass[t])
        self._default_inertia = m.body_inertia[t].copy()
        self._default_armature = m.dof_armature[self.hinge_dofs].copy()
        self.foot_friction = np.zeros(n)
        self.encoder_bias = np.zeros((n, ACT_DIM))
        self.mass_scale = np.zeros(n)
        self.imu_rot = np.zeros((n, 3, 3))
        self.trunk_com = np.zeros((n, 3))
        self.armature_scale = np.ones((n, len(self.hinge_dofs)))
        self.push_left = np.zeros(n)
        for i, rng in enumerate(self._dr_rngs):
            self.foot_friction[i] = rng.uniform(*self.dr_foot_friction)
            self.encoder_bias[i] = rng.uniform(-ENCODER_BIAS, ENCODER_BIAS, ACT_DIM)
            self.mass_scale[i] = rng.uniform(*MASS_SCALE)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-8
            self.imu_rot[i] = axis_angle_matrix(
                axis, rng.uniform(0.0, math.radians(IMU_MAX_ANGLE_DEG))
            )

    def _sample_dr_reset(self, i: int) -> None:
        rng = self._dr_rngs[i]
        self.trunk_com[i] = rng.uniform(-TRUNK_COM, TRUNK_COM, 3)
        self.armature_scale[i] = rng.uniform(*ARMATURE_SCALE, len(self.hinge_dofs))
        self.push_left[i] = rng.uniform(*PUSH_INTERVAL_S)

    def _apply_physics(self, i: int) -> None:
        if not self.dr:
            return
        m, t = self.model, self.trunk_id
        m.geom_friction[self.foot_geoms, 0] = self.foot_friction[i]
        m.body_ipos[t] = self._default_ipos + self.trunk_com[i]
        m.body_mass[t] = self._default_mass * self.mass_scale[i]
        m.body_inertia[t] = self._default_inertia * self.mass_scale[i]
        m.dof_armature[self.hinge_dofs] = self._default_armature * self.armature_scale[i]

    def _push(self) -> None:
        self.push_left -= STEP_DT
        for i in np.flatnonzero(self.push_left < 1e-6):
            rng = self._dr_rngs[int(i)]
            self.datas[int(i)].qvel[0:2] += rng.uniform(-PUSH_DV, PUSH_DV, 2)
            self.push_left[i] = rng.uniform(*PUSH_INTERVAL_S)

    # ----- commands -------------------------------------------------------------

    @property
    def obs(self) -> np.ndarray:
        return self._obs.copy()

    @property
    def critic_obs(self) -> np.ndarray:
        return self._critic.copy()

    def _sample_twist(self, i: int) -> None:
        rng = self._rngs[i]
        twist = rng.uniform(-TWIST_RANGE, TWIST_RANGE)
        r = rng.uniform()
        if r < FORWARD_FRACTION:
            twist = np.array([max(abs(twist[0]), FORWARD_MIN_VX), 0.0, 0.0])
        elif r < FORWARD_FRACTION + TURN_FRACTION:
            twist = np.array([0.0, 0.0, rng.choice([-1.0, 1.0]) * rng.uniform(*TURN_WZ)])
        if rng.uniform() < STANDING_FRACTION:
            twist[:] = 0.0
        self.twist[i] = twist
        self.twist_left[i] = rng.uniform(*TWIST_RESAMPLE_S)

    def _sample_lag(self, i: int) -> None:
        self.lag[i] = self._rngs[i].integers(0, 2, 2).astype(bool) if self.noise else False
        self.lag_left[i] = LAG_RESAMPLE_STEPS

    def freeze_commands(self, twist: np.ndarray) -> None:
        """Hold a fixed (vx, vy, wz) command for evaluation, also after resets."""
        self._frozen = np.asarray(twist, dtype=float).reshape(3)
        self.twist[:] = self._frozen
        self.twist_left[:] = np.inf
        self._observe_into(np.arange(self.num_envs))

    def _update_commands(self) -> None:
        self.twist_left -= STEP_DT
        for i in np.flatnonzero(self.twist_left <= 0.0):
            self._sample_twist(int(i))
        self.lag_left -= 1
        for i in np.flatnonzero(self.lag_left <= 0):
            self._sample_lag(int(i))

    # ----- reset ----------------------------------------------------------------

    def _reset_env(self, i: int) -> None:
        m, d, rng = self.model, self.datas[i], self._rngs[i]
        if self.dr:
            self._sample_dr_reset(i)
            self._apply_physics(i)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.stand_qpos
        x, y = rng.uniform(-RESET_XY, RESET_XY, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        d.qpos[0:3] = [x, y, self.stand_height + rng.uniform(*RESET_Z_LIFT)]
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self.qpos_idx] += rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, ACT_DIM)
        d.qvel[:] = 0.0
        d.ctrl[:] = self.default_pose
        mujoco.mj_forward(m, d)
        self.episode_steps[i] = 0
        self.last_action[i] = 0.0
        self.last_target[i] = self.default_pose
        self.action_history[:, i] = 0.0
        self.delay[i] = self.action_delay
        if self.dr:
            self.delay[i] += int(self._delay_rngs[i].integers(0, self.dr_delay_max + 1))
        self.air_time[i] = 0.0
        if self._frozen is None:
            self._sample_twist(i)
        else:
            self.twist[i] = self._frozen
            self.twist_left[i] = np.inf
        self._sample_lag(i)
        quat = d.xquat[self.trunk_id][None]
        self.prev_gyro[i] = d.sensordata[self.gyro_adr : self.gyro_adr + 3]
        self.prev_gravity[i] = rotate_inverse(quat, np.array([[0.0, 0.0, -1.0]]))[0]
        self.prev_joint_vel[i] = d.qvel[self.qvel_idx]
        self.contact[i] = self._contacts(d)[0]

    # ----- state ----------------------------------------------------------------

    def _contacts(self, d: mujoco.MjData) -> tuple[np.ndarray, bool]:
        """Feet touching the floor, and whether the two feet touch each other."""
        geom = d.contact.geom[: d.ncon]
        floor = (geom == self.floor_geom).any(axis=1)
        on = [(geom == g).any(axis=1) for g in self.foot_geoms]
        feet = np.array([bool((floor & f).any()) for f in on])
        return feet, bool((on[0] & on[1]).any())

    def _state(self) -> dict[str, np.ndarray]:
        ds, m = self.datas, self.model
        quat = np.array([d.xquat[self.trunk_id] for d in ds])
        root_vel = np.array([d.qvel[0:6] for d in ds])
        feet_vel = np.zeros((len(ds), 2, 6))
        for k, d in enumerate(ds):
            for f, sid in enumerate(self.site_ids):
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, feet_vel[k, f], 0)
        contacts = [self._contacts(d) for d in ds]
        down = np.broadcast_to([0.0, 0.0, -1.0], (len(ds), 3))
        g = self.gyro_adr
        return {
            "gyro": np.array([d.sensordata[g : g + 3] for d in ds]),
            "gravity": rotate_inverse(quat, down),
            "joint_pos": np.array([d.qpos[self.qpos_idx] for d in ds]),
            "joint_vel": np.array([d.qvel[self.qvel_idx] for d in ds]),
            "lin_vel_b": rotate_inverse(quat, root_vel[:, 0:3]),
            "ang_vel_b": root_vel[:, 3:6],
            "ang_vel_w": rotate(quat, root_vel[:, 3:6]),
            "height": np.array([d.qpos[2] for d in ds]),
            "foot_height": np.array([[d.site_xpos[s][2] for s in self.site_ids] for d in ds]),
            "foot_vel_xy": feet_vel[:, :, 3:5],
            "contact": np.array([c[0] for c in contacts]),
            "feet_touch": np.array([c[1] for c in contacts], dtype=float),
            "torque": np.abs(np.array([d.actuator_force for d in ds])),
            "finite": np.array(
                [bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) for d in ds]
            ),
        }

    def _observe_into(self, envs: np.ndarray, s: dict[str, np.ndarray] | None = None) -> None:
        if s is None:
            s = {k: v[envs] for k, v in self._state().items()}
        cmd = self.twist[envs]
        joint_rel = s["joint_pos"] - self.default_pose
        clean = [s["gyro"], s["gravity"], joint_rel, s["joint_vel"], self.last_action[envs], cmd]
        gyro = np.where(self.lag[envs, 0:1], self.prev_gyro[envs], s["gyro"])
        gravity = np.where(self.lag[envs, 1:2], self.prev_gravity[envs], s["gravity"])
        joint_vel = self.prev_joint_vel[envs] if self.noise else s["joint_vel"]
        actor_joint = joint_rel
        if self.dr:
            gyro = np.einsum("nij,nj->ni", self.imu_rot[envs], gyro)
            gravity = np.einsum("nij,nj->ni", self.imu_rot[envs], gravity)
            actor_joint = joint_rel + self.encoder_bias[envs]
        actor = np.concatenate(
            [gyro, gravity, actor_joint, joint_vel, self.last_action[envs], cmd], axis=1
        )
        if self.noise:
            half = NOISE_HALF_WIDTH
            actor[:, : half.size] += np.stack([self._rngs[i].uniform(-half, half) for i in envs])
        extra = [s["lin_vel_b"], s["foot_height"], self.air_time[envs], s["contact"].astype(float)]
        self._obs[envs] = actor.astype(np.float32)
        self._critic[envs] = np.concatenate(clean + extra, axis=1).astype(np.float32)
        self.prev_gyro[envs] = s["gyro"]
        self.prev_gravity[envs] = s["gravity"]
        self.prev_joint_vel[envs] = s["joint_vel"]

    # ----- step -----------------------------------------------------------------

    def _rewards(
        self, s: dict[str, np.ndarray], actions: np.ndarray, cut: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Weighted reward terms; ``cut`` is the target step removed by the slew limit."""
        if cut is None:
            cut = np.zeros_like(actions)
        twist = self.twist
        active = np.linalg.norm(twist[:, :2], axis=1) + np.abs(twist[:, 2]) > COMMAND_THRESHOLD
        lin, ang = s["lin_vel_b"], s["ang_vel_b"]
        lin_err = np.sum(np.square(twist[:, :2] - lin[:, :2]), axis=1) + np.square(lin[:, 2])
        ang_err = np.square(twist[:, 2] - ang[:, 2]) + np.sum(np.square(ang[:, :2]), axis=1)
        q = s["joint_pos"]
        std = np.where(active[:, None], POSE_STD_WALKING, POSE_STD_STANDING)
        out = np.clip(self.soft_lo - q, 0, None) + np.clip(q - self.soft_hi, 0, None)
        air = (self.air_time > AIR_TIME_WINDOW[0]) & (self.air_time < AIR_TIME_WINDOW[1])
        foot_speed = np.linalg.norm(s["foot_vel_xy"], axis=2)
        clearance = s["foot_height"] - s["foot_height"].min(axis=1, keepdims=True)
        values = {
            "track_linear_velocity": np.exp(-lin_err / TRACK_LIN_STD2),
            "track_angular_velocity": np.exp(-ang_err / TRACK_ANG_STD2),
            "upright": np.exp(-np.sum(np.square(s["gravity"][:, :2]), axis=1) / UPRIGHT_STD2),
            "pose": np.exp(-np.mean(np.square(q - self.default_pose) / np.square(std), axis=1)),
            "body_ang_vel": np.sum(np.square(s["ang_vel_w"][:, :2]), axis=1),
            "dof_pos_limits": np.sum(out, axis=1),
            "action_rate_l2": np.sum(np.square(actions - self.last_action), axis=1),
            # Single-support time only: both feet in the air is not a step.
            "air_time": np.sum(air, axis=1) * s["contact"].any(axis=1) * active,
            "foot_clearance": np.sum(np.abs(clearance - FOOT_TARGET_HEIGHT) * foot_speed, axis=1)
            * active,
            "foot_slip": np.sum(np.square(foot_speed) * s["contact"], axis=1),
            "feet_collision": s["feet_touch"],
            "action_slew": np.sum(np.square(cut), axis=1),
        }
        weights = dict(WEIGHTS)
        weights["action_rate_l2"] = stage_value(ACTION_RATE_STAGES, self.common_step)
        return {k: (weights[k] * v * STEP_DT).astype(np.float32) for k, v in values.items()}

    def step(self, actions: np.ndarray) -> BipedStepResult:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, ACT_DIM):
            raise ValueError(f"actions must have shape {(self.num_envs, ACT_DIM)}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        applied = actions
        if len(self.action_history) > 1:
            self.action_history = np.roll(self.action_history, 1, axis=0)
            self.action_history[0] = actions
            applied = self.action_history[self.delay, np.arange(self.num_envs)]
        wanted = self.default_pose + ACTION_SCALE * applied
        targets = wanted
        if self.target_slew is not None:
            step = np.clip(wanted - self.last_target, -self.target_slew, self.target_slew)
            targets = self.last_target + step
        cut = wanted - targets
        slew_limited = np.any(np.abs(cut) > 1e-9, axis=1)
        self.last_target = targets
        for i, d in enumerate(self.datas):
            self._apply_physics(i)
            d.ctrl[:] = targets[i]
            mujoco.mj_step(self.model, d, nstep=DECIMATION)
        self.episode_steps += 1
        self.common_step += 1
        s = self._state()
        finite = s["finite"]
        for value in s.values():
            if value.dtype.kind == "f":
                value[~finite] = 0.0
        contact = s["contact"]
        self.air_time = np.where(contact, 0.0, self.air_time + STEP_DT)
        terms = self._rewards(s, actions, cut)
        self.contact = contact

        tilted = s["gravity"][:, 2] > -math.cos(math.radians(FALL_TILT_DEG))
        low = s["height"] < FALL_HEIGHT_FRACTION * self.stand_height
        terminated = ~finite | tilted | low
        truncated = ~terminated & (self.episode_steps >= self.max_episode_steps)
        reward = np.sum(np.stack(list(terms.values())), axis=0).astype(np.float32)
        reward[~finite] = 0.0
        self.last_action[:] = actions
        torque = s["torque"].astype(np.float32)

        self._update_commands()
        envs = np.arange(self.num_envs)
        self._observe_into(envs, s)
        final_obs = self._obs.copy()
        done = np.flatnonzero(terminated | truncated)
        for j in done:
            self._reset_env(int(j))
        if done.size:
            self._observe_into(done)
        if self.dr:
            self._push()
        return BipedStepResult(
            self.obs,
            self.critic_obs,
            reward,
            terminated,
            truncated,
            final_obs,
            terms,
            torque,
            slew_limited,
        )
