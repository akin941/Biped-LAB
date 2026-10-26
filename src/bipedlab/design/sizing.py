"""Engineering sizing for our biped (WP4-01): mass, CoM, joint loads, power, battery.

Joint loads come from two sources:

- **Static inverse dynamics** on any free-base MJCF (``static_loads``): with the
  robot still, the joint torques are ``tau = g(q) - J^T W``, where ``g`` is the
  gravity load and ``W`` the wrench (force and moment) under the stance foot or
  feet. The base rows of the same equation (``0 = g_base - J_base^T W``) fix ``W``:
  uniquely for one foot, minimum-norm for two. This also gives the centre of
  pressure (CoP) under each foot, which shows whether the pose can be held.
- **Walking telemetry from simulation** (``walking_loads``): peak and RMS torque and
  speed per joint from a recording of a trained policy. We have no policy for our
  robot yet, so the official Microduck walking policy is the reference: its
  walking peaks divided by its own static single-support torques give a dynamic
  factor per joint type, which is applied to our robot's static torques.

All results are plain dicts so the script can print and store them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from bipedlab.design.biped import BipedParams

G = 9.81
JOINT_TYPES = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")


def joint_type(name: str) -> str | None:
    """Map a joint name of either robot to one of ``JOINT_TYPES``."""
    for t in JOINT_TYPES:
        if t in name:
            return t
    return None


def mass_budget(p: BipedParams) -> dict[str, float]:
    sv = p.servo
    rows = {
        "servos (10)": 10 * sv.mass,
        "trunk structure": p.trunk_structure_mass,
        "electronics": p.electronics_mass,
        "battery": p.battery_mass,
        "hip brackets (4)": 4 * p.bracket_mass,
        "thighs (2)": 2 * p.thigh_mass,
        "shins (2)": 2 * p.shin_mass,
        "feet (2)": 2 * p.foot_mass,
    }
    rows["total"] = sum(rows.values())
    return rows


@dataclass
class StaticCase:
    name: str
    qpos: np.ndarray  # full qpos (free base + joints)
    stance_sites: Sequence[str]  # sole sites carrying the weight


def _site_jacobian(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> np.ndarray:
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, model.site(site).id)
    return np.vstack([jp, jr])  # (6, nv): force rows then moment rows


def static_loads(model: mujoco.MjModel, case: StaticCase) -> dict[str, Any]:
    """Joint torques (actuator order) and per-foot CoP for a still pose."""
    data = mujoco.MjData(model)
    data.qpos[:] = case.qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    gravity = data.qfrc_bias.copy()  # qvel = 0: gravity only
    jac = np.vstack([_site_jacobian(model, data, s) for s in case.stance_sites])
    base = slice(0, 6)
    # 0 = g_base - J_base^T W  ->  W (minimum norm when two feet share the load)
    wrench = np.linalg.lstsq(jac[:, base].T, gravity[base], rcond=None)[0]
    tau = gravity - jac.T @ wrench
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    cops = {}
    for k, site in enumerate(case.stance_sites):
        f, mom = wrench[6 * k : 6 * k + 3], wrench[6 * k + 3 : 6 * k + 6]
        # CoP relative to the site, world frame: moment about the site from Fz at r.
        cop = np.array([-mom[1], mom[0]]) / f[2] if f[2] > 1e-9 else np.full(2, np.nan)
        cops[site] = {"fz_n": float(f[2]), "cop_xy_m": cop.tolist()}
    names = [model.joint(j).name for j in model.actuator_trnid[:, 0]]
    return {
        "case": case.name,
        "base_residual": float(np.abs(tau[base]).max()),
        "torque_nm": dict(zip(names, tau[dofs].tolist(), strict=True)),
        "feet": cops,
    }


def by_type(values: dict[str, float]) -> dict[str, float]:
    """Largest |value| per joint type."""
    out: dict[str, float] = {}
    for name, v in values.items():
        t = joint_type(name)
        if t is not None:
            out[t] = max(out.get(t, 0.0), abs(v))
    return out


def walking_loads(arrays: dict[str, np.ndarray], names: Sequence[str],
                  settle_s: float = 1.0) -> dict[str, Any]:  # fmt: skip
    """Peak and RMS |torque| and peak |speed| per joint from a telemetry recording."""
    keep = arrays["time"] >= arrays["time"][0] + settle_s
    tau = arrays["joint_torque"][keep]
    vel = arrays["joint_vel"][keep]
    power = np.abs(tau * vel).sum(axis=1)
    per_joint = {
        n: {
            "peak_torque_nm": float(np.abs(tau[:, i]).max()),
            "rms_torque_nm": float(np.sqrt(np.mean(tau[:, i] ** 2))),
            "peak_speed_rad_s": float(np.abs(vel[:, i]).max()),
        }
        for i, n in enumerate(names)
    }
    return {
        "joints": per_joint,
        "mech_power_mean_w": float(power.mean()),
        "mech_power_peak_w": float(power.max()),
    }


def battery_runtime_min(capacity_mah: float, voltage_v: float, load_w: float,
                        usable: float = 0.8) -> float:  # fmt: skip
    """Minutes of runtime at ``load_w`` using ``usable`` of the pack's energy."""
    return capacity_mah / 1000 * voltage_v * usable / load_w * 60
