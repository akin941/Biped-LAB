"""Batched CPU Microduck environment with the upstream 61/14 contract.

``MicroduckBatchEnv`` steps ``num_envs`` independent ``MjData`` copies of one model at
the policy rate (50 Hz, 4 physics steps of 5 ms) and returns NumPy arrays, so any PPO
library can wrap it without this module importing torch. Observations come from the
same ``ObservationBuilder`` the ONNX evaluation uses; actions map to MJCF position
targets as ``DEFAULT_POSE + action`` (upstream's simulator, no post-processing).

The task is a placeholder "stand still" task for throughput measurements (WP3-01):
alive bonus, an upright term and an action-rate penalty, with a fall ending the
episode. WP3-02 replaces it with upstream's reward terms and command sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from bipedlab.sim import microduck_policy as mp
from bipedlab.sim.rollout import FALL_TILT_DEG

FALL_GRAVITY_Z = -float(np.cos(np.radians(FALL_TILT_DEG)))
RESET_NOISE_RAD = 0.05
ACTION_RATE_WEIGHT = 0.05


@dataclass
class StepResult:
    obs: np.ndarray  # (N, 61) float32, after auto-reset for finished envs
    reward: np.ndarray  # (N,) float32
    terminated: np.ndarray  # (N,) bool: fell
    truncated: np.ndarray  # (N,) bool: reached the episode time limit
    final_obs: np.ndarray  # (N, 61) float32, observation before auto-reset


@dataclass
class MicroduckBatchEnv:
    scene: str | Path
    num_envs: int
    episode_s: float = 10.0
    seed: int = 0
    # Global index of this batch's first env. Each env draws reset noise from its own
    # stream seeded by (seed, global index), so a batch split into shards (see
    # ``parallel_env``) produces exactly the same episodes as one serial batch.
    first_env: int = 0
    model: mujoco.MjModel = field(init=False, repr=False)
    datas: list[mujoco.MjData] = field(init=False, repr=False)
    episode_steps: np.ndarray = field(init=False, repr=False)
    last_action: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.model, _ = mp.load_model_for_policy(self.scene)
        self.observe = mp.ObservationBuilder(self.model)
        self.max_episode_steps = round(self.episode_s * mp.CONTROL_HZ)
        self.command = np.zeros(mp.CMD_DIM, dtype=np.float32)
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_envs)]
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.last_action = np.zeros((self.num_envs, mp.ACT_DIM), dtype=np.float32)
        self._rngs = [
            np.random.default_rng([self.seed, self.first_env + i]) for i in range(self.num_envs)
        ]
        self._obs = np.zeros((self.num_envs, mp.OBS_DIM), dtype=np.float32)
        for i in range(self.num_envs):
            self._reset_env(i)

    @property
    def obs(self) -> np.ndarray:
        return self._obs.copy()

    def _reset_env(self, i: int) -> None:
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)
        mp.upstream_init(self.model, d)
        noise = self._rngs[i].uniform(-RESET_NOISE_RAD, RESET_NOISE_RAD, mp.ACT_DIM)
        d.qpos[self.observe.qpos_idx] += noise
        mujoco.mj_forward(self.model, d)
        self.episode_steps[i] = 0
        self.last_action[i] = 0.0
        self._obs[i] = self.observe(d, self.last_action[i], self.command)

    def _observe_all(self) -> np.ndarray:
        """Batched ``ObservationBuilder`` (same arithmetic, so bitwise-equal rows)."""
        o = self.observe
        gyro = np.array([d.sensordata[o.gyro_adr : o.gyro_adr + 3] for d in self.datas])
        quat = np.array([d.xquat[o.trunk_id] for d in self.datas])
        qpos = np.array([d.qpos[o.qpos_idx] for d in self.datas])
        qvel = np.array([d.qvel[o.qvel_idx] for d in self.datas])
        w, xyz = quat[:, :1], quat[:, 1:4]
        down = np.array([0.0, 0.0, -1.0])
        t = np.cross(xyz, down) * 2.0
        gravity = down - w * t + np.cross(xyz, t)
        command = np.broadcast_to(self.command, (self.num_envs, mp.CMD_DIM))
        parts = [gyro, gravity, qpos - mp.DEFAULT_POSE, qvel, self.last_action, command]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, mp.ACT_DIM):
            raise ValueError(f"actions must have shape {(self.num_envs, mp.ACT_DIM)}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        for i, d in enumerate(self.datas):
            d.ctrl[:] = mp.DEFAULT_POSE + actions[i]
            mujoco.mj_step(self.model, d, nstep=mp.DECIMATION)
        rate = np.sum(np.square(actions - self.last_action), axis=1)
        self.last_action[:] = actions
        obs = self._observe_all()
        gravity_z = obs[:, mp.OBS_SLICES["projected_gravity"].start + 2].astype(np.float64)
        finite = np.all(np.isfinite(obs), axis=1)
        terminated = ~finite | (gravity_z > FALL_GRAVITY_Z)
        reward = (1.0 - gravity_z - ACTION_RATE_WEIGHT * rate.astype(np.float64)).astype(np.float32)
        reward[~finite] = 0.0
        obs[~finite] = 0.0
        self._obs[:] = obs
        self.episode_steps += 1
        truncated = ~terminated & (self.episode_steps >= self.max_episode_steps)
        final_obs = self._obs.copy()
        for done in np.flatnonzero(terminated | truncated):
            self._reset_env(int(done))
        return StepResult(self.obs, reward, terminated, truncated, final_obs)
