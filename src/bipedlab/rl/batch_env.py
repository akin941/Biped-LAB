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

    def step(self, actions: np.ndarray) -> StepResult:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, mp.ACT_DIM):
            raise ValueError(f"actions must have shape {(self.num_envs, mp.ACT_DIM)}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        reward = np.empty(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        for i, d in enumerate(self.datas):
            d.ctrl[:] = mp.DEFAULT_POSE + actions[i]
            for _ in range(mp.DECIMATION):
                mujoco.mj_step(self.model, d)
            rate = float(np.sum(np.square(actions[i] - self.last_action[i])))
            self.last_action[i] = actions[i]
            obs = self.observe(d, self.last_action[i], self.command)
            gravity_z = float(obs[mp.OBS_SLICES["projected_gravity"]][2])
            finite = bool(np.all(np.isfinite(obs)))
            terminated[i] = (not finite) or gravity_z > FALL_GRAVITY_Z
            reward[i] = 1.0 - gravity_z - ACTION_RATE_WEIGHT * rate if finite else 0.0
            self._obs[i] = obs if finite else 0.0
        self.episode_steps += 1
        truncated = ~terminated & (self.episode_steps >= self.max_episode_steps)
        final_obs = self._obs.copy()
        for done in np.flatnonzero(terminated | truncated):
            self._reset_env(int(done))
        return StepResult(self.obs, reward, terminated, truncated, final_obs)
