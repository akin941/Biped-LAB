"""Sit down / stand up / power off: open-loop pose ramps on a biped model (WP5-19).

A PWM servo cannot be told to go soft, and switching it off drops the robot. This
measures, with the model's position servos and no policy:

* ``ramp``: the robot follows a smoothstep from the stand pose (knee 0.6) to a crouch
  (knee ``K``, trunk upright, feet flat) over ``T`` seconds, holds, and comes back.
  Reported: did it stay upright, its lowest trunk height, per-joint peak torque.
* ``drop``: all servos switch off (zero torque) in the stand pose and in each crouch.
  Reported: how far the trunk falls and how fast it hits.

Usage: uv run python scripts/sit_stand.py [--model biped_ds3218b_heavy] [--out r.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab import REPO_ROOT
from bipedlab.design.biped import LEG_JOINTS, crouch_pose

SCENES = REPO_ROOT / "simulation/custom-biped"
STAND_KNEE = 0.6
UPRIGHT_COS = np.cos(0.5)  # trunk tilt beyond 0.5 rad counts as a fall


def _pose(knee: float) -> np.ndarray:
    p = crouch_pose(knee)
    return np.array([p[j] for j in LEG_JOINTS] * 2)


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _load(model_name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(SCENES / f"{model_name}.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    return model, data


def _upright(data: mujoco.MjData) -> bool:
    # z axis of the trunk (body 1) against the world z axis.
    return bool(data.xmat[1].reshape(3, 3)[2, 2] > UPRIGHT_COS)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def ramp(model_name: str, knee: float, seconds: float, hold: float = 1.0) -> dict[str, Any]:
    model, data = _load(model_name)
    _settle(model, data, 0.5)
    a, b = _pose(STAND_KNEE), _pose(knee)
    dt = model.opt.timestep
    peak = np.zeros(model.nu)
    low, upright = float(data.qpos[2]), True
    phases = ((a, b, seconds), (b, b, hold), (b, a, seconds), (a, a, hold))
    for start, end, dur in phases:
        for k in range(int(dur / dt)):
            data.ctrl[:] = start + (end - start) * _smooth((k + 1) * dt / dur)
            mujoco.mj_step(model, data)
            peak = np.maximum(peak, np.abs(data.actuator_force))
            low = min(low, float(data.qpos[2]))
            upright &= _upright(data)
    by_joint = {j: round(float(max(peak[i], peak[i + 5])), 3) for i, j in enumerate(LEG_JOINTS)}
    return {
        "knee": knee,
        "seconds": seconds,
        "upright": upright,
        "back_to_stand": bool(abs(data.qpos[2] - _load(model_name)[1].qpos[2]) < 0.01),
        "lowest_trunk_m": round(low, 4),
        "peak_torque_nm": by_joint,
        "torque_limit_nm": round(float(model.actuator_forcerange[0, 1]), 3),
    }


def drop(model_name: str, knee: float, friction: float | None = None,
         ramp_s: float = 2.0) -> dict[str, Any]:  # fmt: skip
    """Crouch to ``knee`` slowly, then switch every servo off and watch the trunk fall.

    The walking model lets only the feet touch the floor; here every robot geom does, so
    the fall ends on the floor. ``friction``: joint friction (N*m) of an unpowered servo's
    gearbox, unknown until the bench; ``None`` keeps the model's.
    """
    model, data = _load(model_name)
    floor = model.geom("floor").id
    for g in range(model.ngeom):
        if g != floor:
            model.geom_conaffinity[g] |= 1
    model.body_conaffinity[1:] |= 1  # the broadphase filters on the compiled body masks
    _settle(model, data, 0.5)
    a, b = _pose(STAND_KNEE), _pose(knee)
    dt = model.opt.timestep
    for k in range(int(ramp_s / dt)):
        data.ctrl[:] = a + (b - a) * _smooth((k + 1) * dt / ramp_s)
        mujoco.mj_step(model, data)
    _settle(model, data, 0.5)
    z0 = float(data.qpos[2])
    model.actuator_gainprm[:, 0] = 0.0  # servos off: no torque at all
    model.actuator_biasprm[:, 1:3] = 0.0
    if friction is not None:
        model.dof_frictionloss[6:] = friction
    low, vmax = z0, 0.0
    for _ in range(int(1.5 / dt)):
        mujoco.mj_step(model, data)
        low = min(low, float(data.qpos[2]))
        vmax = max(vmax, float(-data.qvel[2]))
    return {
        "knee": knee,
        "gearbox_friction_nm": round(float(model.dof_frictionloss[6]), 3),
        "trunk_before_m": round(z0, 4),
        "trunk_after_m": round(low, 4),
        "fall_m": round(z0 - low, 4),
        "peak_down_speed_mps": round(vmax, 3),
        "upright_after": _upright(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="biped_ds3218b_heavy")
    parser.add_argument("--knees", type=float, nargs="+", default=[1.2, 1.6, 2.0])
    parser.add_argument("--seconds", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--friction", type=float, nargs="*", default=[0.1, 0.2],
                        help="unpowered gearbox friction to try (N*m)")  # fmt: skip
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {
        "model": args.model,
        "env": "open-loop pose ramps on the model's position servos (no policy)",
        "ramp": [ramp(args.model, k, s) for k in args.knees for s in args.seconds],
        "drop": [
            drop(args.model, k, f)
            for f in (None, *args.friction)
            for k in [STAND_KNEE, *args.knees]
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
