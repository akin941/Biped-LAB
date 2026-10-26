"""Load MJCF models and describe them as plain data (for reports and tests)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass(frozen=True)
class JointInfo:
    name: str
    type: str
    range: tuple[float, float] | None
    damping: float
    frictionloss: float
    armature: float


@dataclass(frozen=True)
class ActuatorInfo:
    name: str
    joint: str
    kp: float
    kv: float
    forcerange: tuple[float, float] | None
    ctrlrange: tuple[float, float] | None


@dataclass(frozen=True)
class ModelInfo:
    path: str
    timestep: float
    total_mass: float
    nq: int
    nv: int
    nu: int
    joints: list[JointInfo] = field(default_factory=list)
    actuators: list[ActuatorInfo] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=list)
    keyframes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}


def load(path: str | Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def _range(limited: bool, values: np.ndarray) -> tuple[float, float] | None:
    return (float(values[0]), float(values[1])) if limited else None


def describe(model: mujoco.MjModel, path: str | Path = "") -> ModelInfo:
    joints = []
    for j in range(model.njnt):
        dof = model.jnt_dofadr[j]
        joints.append(
            JointInfo(
                name=model.joint(j).name,
                type=_JOINT_TYPES[int(model.jnt_type[j])],
                range=_range(bool(model.jnt_limited[j]), model.jnt_range[j]),
                damping=float(model.dof_damping[dof]),
                frictionloss=float(model.dof_frictionloss[dof]),
                armature=float(model.dof_armature[dof]),
            )
        )
    actuators = []
    for a in range(model.nu):
        # Position actuators: gain = kp, bias = [0, -kp, -kv].
        actuators.append(
            ActuatorInfo(
                name=model.actuator(a).name,
                joint=model.joint(model.actuator_trnid[a, 0]).name,
                kp=float(model.actuator_gainprm[a, 0]),
                kv=float(-model.actuator_biasprm[a, 2]),
                forcerange=_range(
                    bool(model.actuator_forcelimited[a]), model.actuator_forcerange[a]
                ),
                ctrlrange=_range(bool(model.actuator_ctrllimited[a]), model.actuator_ctrlrange[a]),
            )
        )
    return ModelInfo(
        path=str(path),
        timestep=float(model.opt.timestep),
        total_mass=float(model.body_mass.sum()),
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        joints=joints,
        actuators=actuators,
        sensors=[model.sensor(i).name for i in range(model.nsensor)],
        sites=[model.site(i).name for i in range(model.nsite)],
        keyframes=[model.key(i).name for i in range(model.nkey)],
    )
