"""Upstream Microduck flat velocity task on CPU MuJoCo (WP3-02).

``MicroduckVelocityEnv`` reproduces the task the official ``alpha_walking`` policy was
trained on (``microduck_rl`` ``make_microduck_velocity_env_cfg``, flat terrain, on
mjlab 1.3.0): the same model (``scene_walk.xml``), timing, reset distribution, command
sampling and curricula, actor observation noise and delays, reward terms and
terminations. Every row of ``docs/research/velocity-task-spec.md`` marked **v1** is
implemented here; rows marked **DR** (domain randomization) come with WP3-05.

Known approximations, all recorded in the spec:
- contact, air time and swing peaks are sampled once per control step (20 ms), not
  every 5 ms physics substep as mjlab's contact sensor does;
- self-collisions count robot-robot contacts at the control step, not substeps with
  a contact force above 10 N;
- CPU MuJoCo instead of MuJoCo-Warp (same model, timestep and solver iterations).

Arrays are NumPy (no torch import). Each env draws its randomness from its own stream
seeded by ``(seed, first_env + i)``, so shards in worker processes reproduce a serial
batch exactly. Curricula advance with ``common_step`` (env steps, like mjlab's
``common_step_counter``); a trainer resuming a run sets it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab.sim import bam_xl330
from bipedlab.sim import microduck_policy as mp

STEP_DT = mp.PHYSICS_DT * mp.DECIMATION  # 0.02 s
SOLVER_ITERATIONS = 10
SOLVER_LS_ITERATIONS = 20
EPISODE_S = 20.0
CRITIC_DIM = mp.OBS_DIM + 3 + 2 + 2 + 2  # + base_lin_vel, foot height, air time, contact

# Reset (reset_root_state_uniform).
RESET_Z = (0.12, 0.13)
RESET_XY = 0.5
FALL_TILT_DEG = 70.0

# Actor observation corruption: uniform noise half-widths and delays.
NOISE_GYRO = 0.03
NOISE_GRAVITY = 0.01
NOISE_JOINT_POS = 0.001
NOISE_JOINT_VEL = 0.25
# Covers the first 34 obs entries: gyro, gravity, joint_pos, joint_vel.
NOISE_HALF_WIDTH = np.repeat(
    [NOISE_GYRO, NOISE_GRAVITY, NOISE_JOINT_POS, NOISE_JOINT_VEL], [3, 3, mp.ACT_DIM, mp.ACT_DIM]
)
LAG_RESAMPLE_STEPS = 64  # gyro / gravity lag in {0, 1}, resampled per env

# Commands.
TWIST_RESAMPLE_S = (3.0, 8.0)
POSE_RESAMPLE_S = (2.0, 5.0)
TWIST_RANGE = np.array([0.4, 0.3, 1.0])
FORWARD_FRACTION = 0.20
FORWARD_MIN_VX = 0.3
TURN_FRACTION = 0.15
TURN_WZ = (0.4, 1.0)
BODY_POSE_RANGE = np.array([0.005, 0.005, 0.005, 0.05, 0.05, 0.05])
COMMAND_THRESHOLD = 0.01

# Curricula, keyed on env steps (PPO iterations x 24).
IT = 24
STANDING_STAGES = [(0, 0.02), (500 * IT, 0.05), (750 * IT, 0.1), (1000 * IT, 0.15),
                   (1500 * IT, 0.2), (2000 * IT, 0.25)]  # fmt: skip
HEAD_RANGE_STAGES = [
    (0, (0.05, 0.05, 0.07, 0.015)),
    (500 * IT, (0.17, 0.17, 0.21, 0.047)),
    (1000 * IT, (0.39, 0.39, 0.49, 0.11)),
    (1500 * IT, (0.72, 0.72, 0.91, 0.20)),
    (2000 * IT, (1.10, 1.10, 1.40, 0.31)),
]
ACTION_RATE_STAGES = [(0, -0.1), (500 * IT, -0.2), (750 * IT, -0.4), (1000 * IT, -0.6),
                      (1250 * IT, -0.8), (1500 * IT, -1.0)]  # fmt: skip
HEAD_BIAS_STAGES = [(0, 0.0), (600 * IT, 1.0), (1000 * IT, 2.0), (1500 * IT, 3.0)]

# Reward weights (fixed ones; curriculum weights above).
WEIGHTS = {
    "track_linear_velocity": 2.0,
    "track_angular_velocity": 2.0,
    "upright": 2.0,
    "pose": 1.0,
    "body_ang_vel": -0.05,
    "angular_momentum": -0.02,
    "dof_pos_limits": -1.0,
    "air_time": 3.0,
    "foot_clearance": -2.0,
    "foot_swing_height": -0.25,
    "foot_slip": -0.1,
    "self_collisions": -1.0,
    "head_pose_tracking": 2.0,
}
TRACK_LIN_STD2 = 0.1
TRACK_ANG_STD2 = 0.5
UPRIGHT_STD2 = 0.05
# Leg joints in LEG_JOINTS order: hip_yaw, hip_roll, hip_pitch, knee, ankle (x2).
POSE_STD_STANDING = np.tile([0.1, 0.05, 0.15, 0.15, 0.1], 2)
POSE_STD_WALKING = np.tile([0.3, 0.05, 0.4, 0.4, 0.25], 2)
SOFT_LIMIT_FACTOR = 0.9
AIR_TIME_WINDOW = (0.125, 0.300)
FOOT_TARGET_HEIGHT = 0.02
HEAD_STD = 0.5
HEAD_BIAS_TAU_S = 1.0

FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")


def stage_value(stages: Sequence[tuple[int, Any]], step: int) -> Any:
    """Value of the last stage whose start step is <= ``step``."""
    value = stages[0][1]
    for start, v in stages:
        if step >= start:
            value = v
    return value


def rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate world vectors (N, 3) into the frames of unit quaternions (N, 4) wxyz."""
    w, xyz = quat[:, :1], quat[:, 1:4]
    t = np.cross(xyz, vec) * 2.0
    return np.asarray(vec - w * t + np.cross(xyz, t))


def rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors (N, 3) into the world by quaternions (N, 4) wxyz."""
    w, xyz = quat[:, :1], quat[:, 1:4]
    t = np.cross(xyz, vec) * 2.0
    return np.asarray(vec + w * t + np.cross(xyz, t))


@dataclass
class VelocityStepResult:
    obs: np.ndarray  # (N, 61) float32 actor observation, after auto-reset
    critic_obs: np.ndarray  # (N, 70) float32 privileged observation, after auto-reset
    reward: np.ndarray  # (N,) float32, sum of terms (each weight x value x dt)
    terminated: np.ndarray  # (N,) bool: fell over or non-finite state
    truncated: np.ndarray  # (N,) bool: reached 20 s
    final_obs: np.ndarray  # (N, 61) actor observation before auto-reset
    terms: dict[str, np.ndarray]  # per-term rewards (N,), already x weight x dt


@dataclass
class MicroduckVelocityEnv:
    scene: str | Path
    num_envs: int
    seed: int = 0
    first_env: int = 0
    actuator: str = "mjcf"  # "mjcf" position actuators or "bam" (training's actuator)
    episode_s: float = EPISODE_S
    noise: bool = True  # actor observation noise and delays
    common_step: int = 0
    model: mujoco.MjModel = field(init=False, repr=False)
    datas: list[mujoco.MjData] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.num_envs
        if n < 1:
            raise ValueError("num_envs must be at least 1")
        self.model, _ = mp.load_model_for_policy(self.scene, self.actuator)
        m = self.model
        m.opt.iterations = SOLVER_ITERATIONS
        m.opt.ls_iterations = SOLVER_LS_ITERATIONS
        self.bams: list[bam_xl330.Controller] = []
        if self.actuator == "bam":
            self.bams = [bam_xl330.Controller(m) for _ in range(n)]
            if len(self.bams[0].act_ids) != mp.ACT_DIM:
                raise ValueError("BAM controller does not drive all 14 actuators")
        self.datas = [mujoco.MjData(m) for _ in range(n)]
        self._ids(m)
        self.max_episode_steps = round(self.episode_s / STEP_DT)
        self._rngs = [np.random.default_rng([self.seed, self.first_env + i]) for i in range(n)]

        z = np.zeros
        self.episode_steps = z(n, dtype=np.int64)
        self.last_action = z((n, mp.ACT_DIM), dtype=np.float32)
        self.twist = z((n, 3))
        self.head = z((n, 4))
        self.body = z((n, 6))
        self.twist_left = z(n)
        self.head_left = z(n)
        self.body_left = z(n)
        self.lag = z((n, 2), dtype=bool)  # gyro, gravity
        self.lag_left = z(n, dtype=np.int64)
        self.prev_gyro = z((n, 3))
        self.prev_gravity = z((n, 3))
        self.prev_joint_vel = z((n, mp.ACT_DIM))
        self.air_time = z((n, 2))
        self.contact = z((n, 2), dtype=bool)
        self.peak_height = z((n, 2))
        self.head_bias_ema = z((n, 4))
        self._frozen: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._obs = z((n, mp.OBS_DIM), dtype=np.float32)
        self._critic = z((n, CRITIC_DIM), dtype=np.float32)
        for i in range(n):
            self._reset_env(i)
        self._observe_into(np.arange(n))

    def _ids(self, m: mujoco.MjModel) -> None:
        joints = m.actuator_trnid[:, 0]
        self.qpos_idx = m.jnt_qposadr[joints].astype(int)
        self.qvel_idx = m.jnt_dofadr[joints].astype(int)
        lo, hi = m.jnt_range[joints, 0], m.jnt_range[joints, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_LIMIT_FACTOR
        self.soft_lo, self.soft_hi = mid - half, mid + half

        def sensor_adr(name: str) -> int:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"model lacks sensor {name!r} (use scene_walk.xml)")
            return int(m.sensor_adr[sid])

        self.gyro_adr = sensor_adr("imu_ang_vel")
        self.angmom_adr = sensor_adr("root_angmom")
        self.trunk_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.site_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in FOOT_SITES]
        self.foot_geoms = np.array(
            [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
        )
        self.floor_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if min(self.trunk_id, *self.site_ids, *self.foot_geoms, self.floor_geom) < 0:
            raise ValueError("model lacks trunk_base, foot sites, foot geoms or floor")

    @property
    def obs(self) -> np.ndarray:
        return self._obs.copy()

    @property
    def critic_obs(self) -> np.ndarray:
        return self._critic.copy()

    @property
    def command(self) -> np.ndarray:
        return np.concatenate([self.twist, self.head, self.body], axis=1).astype(np.float32)

    # ----- commands ---------------------------------------------------------------

    def _sample_twist(self, i: int) -> None:
        rng = self._rngs[i]
        twist = rng.uniform(-TWIST_RANGE, TWIST_RANGE)
        r = rng.uniform()
        if r < FORWARD_FRACTION:
            twist = np.array([max(abs(twist[0]), FORWARD_MIN_VX), 0.0, 0.0])
        elif r < FORWARD_FRACTION + TURN_FRACTION:
            twist = np.array([0.0, 0.0, rng.choice([-1.0, 1.0]) * rng.uniform(*TURN_WZ)])
        if rng.uniform() < stage_value(STANDING_STAGES, self.common_step):
            twist[:] = 0.0
        self.twist[i] = twist
        self.twist_left[i] = rng.uniform(*TWIST_RESAMPLE_S)

    def _sample_head(self, i: int) -> None:
        rng = self._rngs[i]
        bound = np.array(stage_value(HEAD_RANGE_STAGES, self.common_step))
        self.head[i] = rng.uniform(-bound, bound)
        self.head_left[i] = rng.uniform(*POSE_RESAMPLE_S)

    def _sample_body(self, i: int) -> None:
        rng = self._rngs[i]
        self.body[i] = rng.uniform(-BODY_POSE_RANGE, BODY_POSE_RANGE)
        self.body_left[i] = rng.uniform(*POSE_RESAMPLE_S)

    def _sample_lag(self, i: int) -> None:
        self.lag[i] = self._rngs[i].integers(0, 2, 2).astype(bool) if self.noise else False
        self.lag_left[i] = LAG_RESAMPLE_STEPS

    def freeze_commands(
        self,
        twist: np.ndarray,
        head: np.ndarray | None = None,
        body: np.ndarray | None = None,
    ) -> None:
        """Hold fixed commands (evaluation): no resampling, also after resets."""
        self._frozen = (
            np.asarray(twist, dtype=float),
            np.zeros(4) if head is None else np.asarray(head, dtype=float),
            np.zeros(6) if body is None else np.asarray(body, dtype=float),
        )
        for i in range(self.num_envs):
            self._apply_frozen(i)
        self._observe_into(np.arange(self.num_envs))

    def _apply_frozen(self, i: int) -> None:
        assert self._frozen is not None
        self.twist[i], self.head[i], self.body[i] = self._frozen
        self.twist_left[i] = self.head_left[i] = self.body_left[i] = np.inf

    def _update_commands(self) -> None:
        for left, sample in (
            (self.twist_left, self._sample_twist),
            (self.head_left, self._sample_head),
            (self.body_left, self._sample_body),
        ):
            left -= STEP_DT
            for i in np.flatnonzero(left <= 0.0):
                sample(int(i))
        self.lag_left -= 1
        for i in np.flatnonzero(self.lag_left <= 0):
            self._sample_lag(int(i))

    # ----- reset ------------------------------------------------------------------

    def _reset_env(self, i: int) -> None:
        m, d, rng = self.model, self.datas[i], self._rngs[i]
        mujoco.mj_resetData(m, d)
        x, y = rng.uniform(-RESET_XY, RESET_XY, 2)
        yaw = rng.uniform(-math.pi, math.pi)
        d.qpos[0:3] = [x, y, rng.uniform(*RESET_Z)]
        d.qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        d.qpos[self.qpos_idx] = mp.DEFAULT_POSE
        d.qvel[:] = 0.0
        if self.bams:
            self.bams[i].reset(d)
            self.bams[i].q_target[:] = mp.DEFAULT_POSE
        else:
            d.ctrl[:] = mp.DEFAULT_POSE
        mujoco.mj_forward(m, d)
        self.episode_steps[i] = 0
        self.last_action[i] = 0.0
        self.air_time[i] = 0.0
        self.peak_height[i] = 0.0
        self.head_bias_ema[i] = 0.0
        self._sample_twist(i)
        self._sample_head(i)
        self._sample_body(i)
        if self._frozen is not None:
            self._apply_frozen(i)
        self._sample_lag(i)
        # Delay buffers start from the reset state.
        quat = d.xquat[self.trunk_id][None]
        self.prev_gyro[i] = d.sensordata[self.gyro_adr : self.gyro_adr + 3]
        self.prev_gravity[i] = rotate_inverse(quat, np.array([[0.0, 0.0, -1.0]]))[0]
        self.prev_joint_vel[i] = d.qvel[self.qvel_idx]
        self.contact[i] = self._foot_contacts(d)[0]

    # ----- state ------------------------------------------------------------------

    def _foot_contacts(self, d: mujoco.MjData) -> tuple[np.ndarray, int]:
        """Feet touching the floor, and the number of robot-robot contacts."""
        geom = d.contact.geom[: d.ncon]
        floor = (geom == self.floor_geom).any(axis=1)
        feet = np.array([bool((floor & (geom == g).any(axis=1)).any()) for g in self.foot_geoms])
        return feet, int((~floor).sum())

    def _state(self, envs: np.ndarray) -> dict[str, np.ndarray]:
        ds = [self.datas[i] for i in envs]
        g, ga = self.gyro_adr, self.angmom_adr
        quat = np.array([d.xquat[self.trunk_id] for d in ds])
        root_vel = np.array([d.qvel[0:6] for d in ds])  # world linear, body angular
        feet_vel = np.zeros((len(ds), 2, 6))
        for k, d in enumerate(ds):
            for f, sid in enumerate(self.site_ids):
                mujoco.mj_objectVelocity(
                    self.model, d, mujoco.mjtObj.mjOBJ_SITE, sid, feet_vel[k, f], 0
                )
        contacts = [self._foot_contacts(d) for d in ds]
        n = len(ds)
        down = np.broadcast_to([0.0, 0.0, -1.0], (n, 3))
        return {
            "gyro": np.array([d.sensordata[g : g + 3] for d in ds]),
            "gravity": rotate_inverse(quat, down),
            "joint_pos": np.array([d.qpos[self.qpos_idx] for d in ds]),
            "joint_vel": np.array([d.qvel[self.qvel_idx] for d in ds]),
            "lin_vel_b": rotate_inverse(quat, root_vel[:, 0:3]),
            "ang_vel_b": root_vel[:, 3:6],
            "ang_vel_w": rotate(quat, root_vel[:, 3:6]),
            "angmom": np.array([d.sensordata[ga : ga + 3] for d in ds]),
            "foot_height": np.array([[d.site_xpos[s][2] for s in self.site_ids] for d in ds]),
            "foot_vel_xy": feet_vel[:, :, 3:5],  # world linear velocity (rot first, then lin)
            "contact": np.array([c[0] for c in contacts]),
            "self_contacts": np.array([c[1] for c in contacts], dtype=float),
            "finite": np.array(
                [bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) for d in ds]
            ),
        }

    def _observe_into(self, envs: np.ndarray, s: dict[str, np.ndarray] | None = None) -> None:
        """Write actor and critic observations of ``envs`` and advance delay buffers."""
        if s is None:
            s = self._state(envs)
        cmd = self.command[envs]
        joint_rel = s["joint_pos"] - mp.DEFAULT_POSE
        clean = [s["gyro"], s["gravity"], joint_rel, s["joint_vel"], self.last_action[envs], cmd]
        gyro = np.where(self.lag[envs, 0:1], self.prev_gyro[envs], s["gyro"])
        gravity = np.where(self.lag[envs, 1:2], self.prev_gravity[envs], s["gravity"])
        joint_vel = self.prev_joint_vel[envs] if self.noise else s["joint_vel"]
        actor = np.concatenate(
            [gyro, gravity, joint_rel, joint_vel, self.last_action[envs], cmd], axis=1
        )
        if self.noise:
            half = NOISE_HALF_WIDTH
            noise = np.stack([self._rngs[i].uniform(-half, half) for i in envs])
            actor[:, : half.size] += noise
        critic_extra = [s["lin_vel_b"], s["foot_height"], self.air_time[envs],
                        s["contact"].astype(float)]  # fmt: skip
        self._obs[envs] = actor.astype(np.float32)
        self._critic[envs] = np.concatenate(clean + critic_extra, axis=1).astype(np.float32)
        self.prev_gyro[envs] = s["gyro"]
        self.prev_gravity[envs] = s["gravity"]
        self.prev_joint_vel[envs] = s["joint_vel"]

    # ----- step -------------------------------------------------------------------

    def _rewards(
        self, s: dict[str, np.ndarray], actions: np.ndarray, first_contact: np.ndarray
    ) -> dict[str, np.ndarray]:
        twist = self.twist
        cmd_total = np.linalg.norm(twist[:, :2], axis=1) + np.abs(twist[:, 2])
        active = cmd_total > COMMAND_THRESHOLD
        lin, ang = s["lin_vel_b"], s["ang_vel_b"]
        lin_err = np.sum(np.square(twist[:, :2] - lin[:, :2]), axis=1) + np.square(lin[:, 2])
        ang_err = np.square(twist[:, 2] - ang[:, 2]) + np.sum(np.square(ang[:, :2]), axis=1)
        legs = s["joint_pos"][:, mp.LEG_JOINTS] - mp.DEFAULT_POSE[mp.LEG_JOINTS]
        std = np.where(active[:, None], POSE_STD_WALKING, POSE_STD_STANDING)
        q = s["joint_pos"]
        out_of_limits = np.clip(self.soft_lo - q, 0, None) + np.clip(q - self.soft_hi, 0, None)
        air = (self.air_time > AIR_TIME_WINDOW[0]) & (self.air_time < AIR_TIME_WINDOW[1])
        foot_speed = np.linalg.norm(s["foot_vel_xy"], axis=2)
        swing_err = np.square(self.peak_height / FOOT_TARGET_HEIGHT - 1.0)
        head_err = s["joint_pos"][:, mp.HEAD_JOINTS] - mp.DEFAULT_POSE[mp.HEAD_JOINTS] - self.head
        alpha = min(1.0, STEP_DT / HEAD_BIAS_TAU_S)
        self.head_bias_ema = (1 - alpha) * self.head_bias_ema + alpha * head_err
        values = {
            "track_linear_velocity": np.exp(-lin_err / TRACK_LIN_STD2),
            "track_angular_velocity": np.exp(-ang_err / TRACK_ANG_STD2),
            "upright": np.exp(-np.sum(np.square(s["gravity"][:, :2]), axis=1) / UPRIGHT_STD2),
            "pose": np.exp(-np.mean(np.square(legs) / np.square(std), axis=1)),
            "body_ang_vel": np.sum(np.square(s["ang_vel_w"][:, :2]), axis=1),
            "angular_momentum": np.sum(np.square(s["angmom"]), axis=1),
            "dof_pos_limits": np.sum(out_of_limits, axis=1),
            "action_rate_l2": np.sum(np.square(actions - self.last_action), axis=1),
            "air_time": np.sum(air, axis=1) * active,
            "foot_clearance": np.sum(
                np.abs(s["foot_height"] - FOOT_TARGET_HEIGHT) * foot_speed, axis=1
            )
            * active,
            "foot_swing_height": np.sum(swing_err * first_contact, axis=1) * active,
            "foot_slip": np.sum(np.square(foot_speed) * s["contact"], axis=1) * active,
            "self_collisions": s["self_contacts"],
            "head_pose_tracking": np.mean(np.exp(-np.square(head_err / HEAD_STD)), axis=1),
            "head_pose_bias": -np.mean(np.abs(self.head_bias_ema), axis=1),
        }
        weights = dict(WEIGHTS)
        weights["action_rate_l2"] = stage_value(ACTION_RATE_STAGES, self.common_step)
        weights["head_pose_bias"] = stage_value(HEAD_BIAS_STAGES, self.common_step)
        return {k: (weights[k] * v * STEP_DT).astype(np.float32) for k, v in values.items()}

    def step(self, actions: np.ndarray) -> VelocityStepResult:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, mp.ACT_DIM):
            raise ValueError(f"actions must have shape {(self.num_envs, mp.ACT_DIM)}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        m = self.model
        targets = mp.DEFAULT_POSE + actions
        for i, d in enumerate(self.datas):
            if self.bams:
                bam = self.bams[i]
                bam.q_target[:] = targets[i]
                for _ in range(mp.DECIMATION):
                    bam.update(d)
                    mujoco.mj_step(m, d)
            else:
                d.ctrl[:] = targets[i]
                mujoco.mj_step(m, d, nstep=mp.DECIMATION)
        self.episode_steps += 1
        self.common_step += 1
        envs = np.arange(self.num_envs)
        s = self._state(envs)
        finite = s["finite"]
        for value in s.values():
            if value.dtype.kind == "f":
                value[~finite] = 0.0

        # Contact bookkeeping (mjlab ContactSensor air time, swing peaks).
        contact = s["contact"]
        first_contact = contact & ~self.contact
        self.air_time = np.where(contact, 0.0, self.air_time + STEP_DT)
        in_air = ~contact
        self.peak_height = np.where(
            in_air, np.maximum(self.peak_height, s["foot_height"]), self.peak_height
        )
        terms = self._rewards(s, actions, first_contact)
        self.peak_height = np.where(first_contact, 0.0, self.peak_height)
        self.contact = contact

        terminated = ~finite | (s["gravity"][:, 2] > -math.cos(math.radians(FALL_TILT_DEG)))
        truncated = ~terminated & (self.episode_steps >= self.max_episode_steps)
        reward = np.sum(np.stack(list(terms.values())), axis=0).astype(np.float32)
        reward[~finite] = 0.0
        self.last_action[:] = actions

        self._update_commands()
        self._observe_into(envs, s)
        final_obs = self._obs.copy()
        done = np.flatnonzero(terminated | truncated)
        for j in done:
            self._reset_env(int(j))
        if done.size:
            self._observe_into(done)
        return VelocityStepResult(
            self.obs, self.critic_obs, reward, terminated, truncated, final_obs, terms
        )
