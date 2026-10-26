"""Deterministic open-loop rollouts with simple stability metrics.

A rollout holds the position actuators at a keyframe's ``ctrl`` (no policy) and
records trunk height and tilt at the control rate. It answers questions such as
"does this robot stand on its own servos?" and "is the simulation deterministic?".
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

# Tilt beyond this (angle between body z-axis and world z-axis) counts as a fall.
FALL_TILT_DEG = 60.0

ControlFn = Callable[[mujoco.MjModel, mujoco.MjData], None]


@dataclass(frozen=True)
class RolloutResult:
    duration_s: float
    control_hz: float
    times: np.ndarray
    trunk_height: np.ndarray
    tilt_deg: np.ndarray
    fell: bool
    fall_time_s: float | None
    finite: bool
    state_digest: str

    def summary(self) -> dict[str, float | bool | str | None]:
        return {
            "duration_s": self.duration_s,
            "control_hz": self.control_hz,
            "fell": self.fell,
            "fall_time_s": self.fall_time_s,
            "min_trunk_height_m": float(self.trunk_height.min()),
            "final_trunk_height_m": float(self.trunk_height[-1]),
            "max_tilt_deg": float(self.tilt_deg.max()),
            "finite": self.finite,
            "state_digest": self.state_digest,
        }


def tilt_deg(quat_wxyz: np.ndarray) -> float:
    """Angle between the body's z-axis and world z-axis, from a unit quaternion."""
    _, x, y, _ = quat_wxyz
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def settle_onto_floor(model: mujoco.MjModel, data: mujoco.MjData, step_m: float = 0.0005) -> None:
    """Lower a free-floating root until the first contact, so it starts on the floor."""
    for _ in range(int(0.2 / step_m)):
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            return
        data.qpos[2] -= step_m
    raise RuntimeError("no contact found within 0.2 m; is there a floor?")


def hold_ctrl(target: np.ndarray) -> ControlFn:
    def control(_: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.ctrl[:] = target

    return control


def rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    keyframe: str | None,
    duration_s: float,
    control_hz: float = 50.0,
    control: ControlFn | None = None,
    settle: bool = True,
    init: ControlFn | None = None,
    substep: ControlFn | None = None,
) -> RolloutResult:
    """Reset to ``keyframe`` and simulate, calling ``control`` at ``control_hz``.

    ``keyframe=None`` resets to the model's default state (for scenes without
    keyframes, e.g. upstream's ball scene); ``init`` then sets the pose.

    The root body must be the first joint and free (qpos[0:7] = pos + quat).
    Without ``control`` the keyframe's ctrl is held. ``init`` may edit the state
    after the keyframe reset (before optional settling). ``substep`` runs before
    every physics step (e.g. an actuator model that recomputes torque each step).
    """
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("rollout expects a free-floating root as the first joint")
    if keyframe is None:
        mujoco.mj_resetData(model, data)
    else:
        mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    if init is not None:
        init(model, data)
        mujoco.mj_forward(model, data)
    if settle:
        settle_onto_floor(model, data)
    control = control or hold_ctrl(data.ctrl.copy())

    substeps = round(1.0 / (control_hz * model.opt.timestep))
    if substeps < 1 or abs(substeps * model.opt.timestep * control_hz - 1.0) > 1e-6:
        raise ValueError(f"control_hz={control_hz} is not a divisor of 1/timestep")
    n = round(duration_s * control_hz)

    times = np.empty(n)
    heights = np.empty(n)
    tilts = np.empty(n)
    fall_time = None
    for i in range(n):
        control(model, data)
        if substep is None:
            mujoco.mj_step(model, data, nstep=substeps)
        else:
            for _ in range(substeps):
                substep(model, data)
                mujoco.mj_step(model, data)
        times[i] = data.time
        heights[i] = data.qpos[2]
        tilts[i] = tilt_deg(data.qpos[3:7])
        if fall_time is None and tilts[i] > FALL_TILT_DEG:
            fall_time = float(data.time)

    finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
    digest = hashlib.sha256(
        np.concatenate([data.qpos, data.qvel]).astype("<f8").tobytes()
    ).hexdigest()[:16]
    return RolloutResult(
        duration_s=duration_s,
        control_hz=control_hz,
        times=times,
        trunk_height=heights,
        tilt_deg=tilts,
        fell=fall_time is not None,
        fall_time_s=fall_time,
        finite=finite,
        state_digest=digest,
    )
