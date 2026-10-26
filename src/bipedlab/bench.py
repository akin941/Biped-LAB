"""Small, repeatable CPU benchmarks. Results are only comparable within one environment."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
    }


def physics_steps_per_s(scene: str | Path, dt: float = 0.005, seconds: float = 2.0) -> float:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data, nstep=100)  # warm-up
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        mujoco.mj_step(model, data, nstep=500)
        steps += 500
    return steps / (time.perf_counter() - start)


def policy_loop_hz(scene: str | Path, policy_path: str | Path, seconds: float = 2.0) -> float:
    """Control steps/s of the full obs -> ONNX -> ctrl -> 4 physics steps loop."""
    from bipedlab.sim import microduck_policy as mp

    model, data = mp.load_model_for_policy(scene)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    ctrl = mp.PolicyController(
        policy=mp.OnnxPolicy(policy_path), observe=mp.ObservationBuilder(model)
    )
    n, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        ctrl(model, data)
        mujoco.mj_step(model, data, nstep=mp.DECIMATION)
        n += 1
        if len(ctrl.log) > 1000:
            ctrl.log.clear()
    return n / (time.perf_counter() - start)


def batch_env_steps_per_s(
    scene: str | Path, num_envs: int, workers: int, seconds: float = 2.0
) -> float:
    """Env steps/s (summed over envs) of the Microduck batch env with random actions.

    ``workers == 0`` times the serial ``MicroduckBatchEnv`` in this process; otherwise
    ``ParallelBatchEnv`` with that many worker processes.
    """
    from bipedlab.rl.batch_env import MicroduckBatchEnv
    from bipedlab.rl.parallel_env import ParallelBatchEnv
    from bipedlab.sim import microduck_policy as mp

    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.2, 0.2, (32, num_envs, mp.ACT_DIM)).astype(np.float32)
    env = MicroduckBatchEnv(scene, num_envs) if workers == 0 else None
    parallel = ParallelBatchEnv(scene, num_envs, workers) if workers else None
    step = env.step if env is not None else parallel.step  # type: ignore[union-attr]
    try:
        step(actions[0])  # warm-up
        n, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            step(actions[n % 32])
            n += 1
        return n * num_envs / (time.perf_counter() - start)
    finally:
        if parallel is not None:
            parallel.close()
