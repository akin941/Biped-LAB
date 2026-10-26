"""``MicroduckBatchEnv`` split into shards that step in worker processes (WP3-06).

The batch env spends most of PPO's wall time stepping MuJoCo one env after another in
one Python process (ADR-0012). ``ParallelBatchEnv`` keeps the same interface but
gives each of ``workers`` processes a contiguous shard of the envs, sends the actions
over a pipe and concatenates the results. Envs are seeded by global index, so for a
fixed seed it returns the same arrays as the serial env.

``env_cls`` selects the env (``MicroduckBatchEnv`` by default, or
``MicroduckVelocityEnv``); extra keyword arguments go to its constructor. Step results
are concatenated field by field (dict fields key by key).

Workers use the ``spawn`` start method on every platform (macOS's default; on Linux
it avoids forking a parent that may hold torch thread pools).
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mpr
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab.rl.batch_env import MicroduckBatchEnv
from bipedlab.sim import microduck_policy as mp


def _worker(conn: Connection, env_cls: type, kwargs: dict[str, Any]) -> None:
    try:
        env = env_cls(**kwargs)
    except Exception as exc:  # report construction errors to the parent
        conn.send(("error", repr(exc)))
        return
    conn.send(("ready", (env.obs, env.max_episode_steps, getattr(env, "critic_obs", None))))
    while True:
        cmd, payload = conn.recv()
        if cmd == "step":
            try:
                out = env.step(payload)
            except Exception as exc:
                conn.send(("error", repr(exc)))
                continue
            conn.send(("ok", (out, env.episode_steps.copy())))
        elif cmd == "close":
            conn.close()
            return


def shard_sizes(num_envs: int, workers: int) -> list[int]:
    """Split ``num_envs`` into ``workers`` contiguous shards differing by at most one."""
    if workers < 1 or num_envs < workers:
        raise ValueError("need 1 <= workers <= num_envs")
    base, extra = divmod(num_envs, workers)
    return [base + (1 if w < extra else 0) for w in range(workers)]


def concat_results(outs: list[Any]) -> Any:
    """Concatenate per-shard step results (dataclasses of arrays / dicts of arrays)."""
    merged: dict[str, Any] = {}
    for f in dataclasses.fields(outs[0]):
        values = [getattr(o, f.name) for o in outs]
        if isinstance(values[0], dict):
            merged[f.name] = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
        else:
            merged[f.name] = np.concatenate(values)
    return type(outs[0])(**merged)


class ParallelBatchEnv:
    def __init__(
        self,
        scene: str | Path,
        num_envs: int,
        workers: int,
        episode_s: float = 10.0,
        seed: int = 0,
        env_cls: type = MicroduckBatchEnv,
        **env_kwargs: Any,
    ) -> None:
        self.num_envs = num_envs
        self.sizes = shard_sizes(num_envs, workers)
        self.bounds = np.cumsum([0, *self.sizes])
        ctx = mpr.get_context("spawn")
        self._conns: list[Connection] = []
        self._procs: list[Any] = []
        for w, size in enumerate(self.sizes):
            parent, child = ctx.Pipe()
            kwargs = {"scene": str(scene), "num_envs": size, "episode_s": episode_s,
                      "seed": seed, "first_env": int(self.bounds[w]), **env_kwargs}  # fmt: skip
            proc = ctx.Process(target=_worker, args=(child, env_cls, kwargs), daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        ready = [self._recv(c) for c in self._conns]
        self._obs = np.concatenate([r[0] for r in ready])
        self.max_episode_steps = int(ready[0][1])
        has_critic = ready[0][2] is not None
        self._critic = np.concatenate([r[2] for r in ready]) if has_critic else None
        self.episode_steps = np.zeros(num_envs, dtype=np.int64)

    def _recv(self, conn: Connection) -> Any:
        status, payload = conn.recv()
        if status == "error":
            self.close()
            raise RuntimeError(f"env worker failed: {payload}")
        return payload

    @property
    def obs(self) -> np.ndarray:
        return self._obs.copy()

    @property
    def critic_obs(self) -> np.ndarray:
        if self._critic is None:
            raise AttributeError("this env has no critic observation")
        return self._critic.copy()

    def step(self, actions: np.ndarray) -> Any:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, mp.ACT_DIM):
            raise ValueError(f"actions must have shape {(self.num_envs, mp.ACT_DIM)}")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        for w, conn in enumerate(self._conns):
            conn.send(("step", actions[self.bounds[w] : self.bounds[w + 1]]))
        parts = [self._recv(c) for c in self._conns]
        outs: list[Any] = [p[0] for p in parts]
        self.episode_steps = np.concatenate([p[1] for p in parts])
        result = concat_results(outs)
        self._obs = result.obs.copy()
        if self._critic is not None:
            self._critic = result.critic_obs.copy()
        return result

    def close(self) -> None:
        for conn in self._conns:
            try:
                conn.send(("close", None))
                conn.close()
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs = [], []

    def __enter__(self) -> ParallelBatchEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
