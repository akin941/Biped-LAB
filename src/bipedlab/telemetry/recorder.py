"""Record what the robot is doing at every control step.

Wrap any rollout control function with :class:`Recorder`; it samples the state
just before the controller acts (so ``ctrl`` is the target chosen at that step)
and stores fixed-shape arrays. ``save()`` writes one ``.npz`` plus a JSON sidecar
describing every channel (name, shape, unit), so reports never guess.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

ControlFn = Callable[[mujoco.MjModel, mujoco.MjData], None]

SCHEMA_VERSION = 1

# name -> (unit, description). Shapes are recorded at save time.
CHANNELS: dict[str, tuple[str, str]] = {
    "time": ("s", "simulation time at the control step"),
    "wall_dt": ("s", "wall-clock time spent since the previous control step"),
    "joint_pos": ("rad", "actuated joint positions, actuator order"),
    "joint_vel": ("rad/s", "actuated joint velocities"),
    "joint_target": ("rad", "position target written by the controller this step"),
    "joint_torque": ("N*m", "actuator force from the previous physics step"),
    "power": ("W", "sum over joints of |torque * velocity| (mechanical effort)"),
    "trunk_pos": ("m", "root position, world frame"),
    "trunk_quat": ("-", "root orientation quaternion w,x,y,z"),
    "tilt": ("deg", "angle between trunk z-axis and world vertical"),
    "gyro": ("rad/s", "root angular velocity, body frame"),
    "lin_vel_body": ("m/s", "root linear velocity, body frame (x forward)"),
    "foot_contact": ("bool", "left, right foot touching anything"),
    "command": ("mixed", "controller command vector (policy-specific)"),
}


@dataclass
class Recorder:
    """Wraps a control function and records telemetry each call."""

    model: mujoco.MjModel
    control: ControlFn
    command: Callable[[], np.ndarray] = field(default=lambda: np.zeros(0))
    foot_bodies: tuple[str, str] = ("left_foot", "right_foot")
    # Joint position targets; defaults to data.ctrl (position actuators). With a
    # torque-level actuator model (BAM) ctrl holds torques, so the runner sets this.
    target: Callable[[], np.ndarray] | None = None
    rows: dict[str, list[np.ndarray]] = field(default_factory=dict)
    _last_wall: float | None = None

    def __post_init__(self) -> None:
        m = self.model
        joints = m.actuator_trnid[:, 0]
        self._qpos = m.jnt_qposadr[joints].astype(int)
        self._qvel = m.jnt_dofadr[joints].astype(int)
        self._feet = [self._body_id(name) for name in self.foot_bodies]
        self.joint_names = [m.joint(int(j)).name for j in joints]
        self.rows = {name: [] for name in CHANNELS}

    def _body_id(self, name: str) -> int:
        for candidate in (name, name.replace("_foot", "_foot_link"), f"{name}_link"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid >= 0:
                return int(bid)
        # Fall back to the body owning a site of that name (Microduck has foot sites).
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return int(self.model.site_bodyid[sid])
        return -1

    def _contacts(self, data: mujoco.MjData) -> np.ndarray:
        touching = np.zeros(len(self._feet), dtype=bool)
        for i in range(data.ncon):
            c = data.contact[i]
            bodies = (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2])
            for k, foot in enumerate(self._feet):
                if foot >= 0 and foot in bodies:
                    touching[k] = True
        return touching

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        now = time.perf_counter()
        wall_dt = 0.0 if self._last_wall is None else now - self._last_wall
        self._last_wall = now

        self.control(model, data)

        quat = data.qpos[3:7].copy()
        flat = np.empty(9)
        mujoco.mju_quat2Mat(flat, quat)
        rot = flat.reshape(3, 3)
        vel = data.qvel[self._qvel]
        torque = data.actuator_force.copy()
        up_z = float(np.clip(rot[2, 2], -1.0, 1.0))
        row = {
            "time": np.array(data.time),
            "wall_dt": np.array(wall_dt),
            "joint_pos": data.qpos[self._qpos].copy(),
            "joint_vel": vel.copy(),
            "joint_target": (data.ctrl if self.target is None else self.target()).copy(),
            "joint_torque": torque,
            "power": np.array(np.abs(torque * vel).sum()),
            "trunk_pos": data.qpos[0:3].copy(),
            "trunk_quat": quat,
            "tilt": np.array(np.degrees(np.arccos(up_z))),
            "gyro": data.qvel[3:6].copy(),  # free joint angular velocity is body frame
            "lin_vel_body": rot.T @ data.qvel[0:3],
            "foot_contact": self._contacts(data),
            "command": np.asarray(self.command(), dtype=np.float64).copy(),
        }
        for name, value in row.items():
            self.rows[name].append(value)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.stack(values) for name, values in self.rows.items() if values}

    def save(self, path: str | Path, meta: dict[str, Any] | None = None) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.arrays()
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "channels": {
                name: {
                    "unit": CHANNELS[name][0],
                    "description": CHANNELS[name][1],
                    "shape": list(arrays[name].shape),
                }
                for name in arrays
            },
            "joint_names": self.joint_names,
            "meta": meta or {},
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
        return path


def load(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path).with_suffix(".npz")
    with np.load(path) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return arrays, json.loads(path.with_suffix(".json").read_text())


def summarize(arrays: dict[str, np.ndarray], fall_tilt_deg: float = 60.0) -> dict[str, Any]:
    """Plain-number summary a person can read without plots."""
    t = arrays["time"]
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    fallen = arrays["tilt"] > fall_tilt_deg
    first_fall = float(t[np.argmax(fallen)]) if fallen.any() else None
    wall = arrays["wall_dt"][1:]
    summary: dict[str, Any] = {
        "steps": int(len(t)),
        "sim_duration_s": duration,
        "fell": bool(fallen.any()),
        "first_fall_s": first_fall,
        "max_tilt_deg": float(arrays["tilt"].max()),
        "mean_power_w": float(arrays["power"].mean()),
        "peak_abs_torque_nm": float(np.abs(arrays["joint_torque"]).max()),
        "mean_vx_body_mps": float(arrays["lin_vel_body"][:, 0].mean()),
        "mean_yaw_rate_rps": float(arrays["gyro"][:, 2].mean()),
        "foot_contact_ratio": arrays["foot_contact"].mean(axis=0).round(3).tolist(),
    }
    if len(wall) and wall.sum() > 0:
        summary["realtime_factor"] = float(duration / wall.sum())
    summary.update(timing(arrays))
    if arrays["command"].shape[-1] >= 3:
        cmd = arrays["command"]
        summary["vx_tracking_error_mps"] = float(
            np.abs(arrays["lin_vel_body"][:, 0] - cmd[:, 0]).mean()
        )
        summary["yaw_tracking_error_rps"] = float(np.abs(arrays["gyro"][:, 2] - cmd[:, 2]).mean())
    return summary


def profile(
    arrays: dict[str, np.ndarray],
    joint_names: list[str] | None = None,
    every_s: float = 0.5,
    torque_limit_nm: float | None = None,
) -> dict[str, Any]:
    """Text-sized view of a recording: trunk state over time and per-joint effort.

    ``timeline`` samples tilt, height, forward speed and the largest |target - pos|
    every ``every_s``. ``joints`` gives, per joint, the target range, the mean
    tracking error and (with ``torque_limit_nm``) the fraction of steps at the limit.
    """
    t = arrays["time"]
    pos, target, torque = arrays["joint_pos"], arrays["joint_target"], arrays["joint_torque"]
    err = target - pos
    period = float(np.median(np.diff(t))) if len(t) > 1 else every_s
    stride = max(1, round(every_s / period))
    quat = arrays["trunk_quat"]
    # Signed forward pitch from the quaternion (w, x, y, z); positive = nose down.
    pitch = np.degrees(
        np.arcsin(np.clip(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
    )
    timeline = [
        {
            "t": round(float(t[i]), 2),
            "tilt_deg": round(float(arrays["tilt"][i]), 1),
            "pitch_deg": round(float(pitch[i]), 1),
            "height_m": round(float(arrays["trunk_pos"][i, 2]), 3),
            "vx_mps": round(float(arrays["lin_vel_body"][i, 0]), 3),
            "worst_joint_err_rad": round(float(np.abs(err[i]).max()), 3),
        }
        for i in range(0, len(t), stride)
    ]
    names = joint_names or [f"j{k}" for k in range(pos.shape[1])]
    joints: dict[str, dict[str, float]] = {}
    for k, name in enumerate(names):
        row = {
            "target_min": round(float(target[:, k].min()), 3),
            "target_max": round(float(target[:, k].max()), 3),
            "pos_min": round(float(pos[:, k].min()), 3),
            "pos_max": round(float(pos[:, k].max()), 3),
            "mean_abs_err": round(float(np.abs(err[:, k]).mean()), 3),
            "peak_abs_torque": round(float(np.abs(torque[:, k]).max()), 3),
        }
        if torque_limit_nm is not None:
            row["at_limit"] = round(
                float((np.abs(torque[:, k]) >= 0.99 * torque_limit_nm).mean()), 3
            )
        joints[name] = row
    return {"timeline": timeline, "joints": joints}


def timing(arrays: dict[str, np.ndarray], control_hz: float = 50.0) -> dict[str, Any]:
    """Control-loop timing (backlog WP2-03).

    ``sim_*`` is the period in simulation time between control steps: it must be
    exactly 1/control_hz, otherwise the decimation is wrong. ``wall_*`` is the
    wall-clock time per step on the machine that ran the recording. The simulator
    is not paced to real time, so wall time is compute cost; ``deadline_miss_ratio``
    is the share of steps that took longer than one control period, i.e. how often
    this machine could not have kept up with a real-time 50 Hz loop.
    """
    period = 1.0 / control_hz
    out: dict[str, Any] = {"control_period_ms": period * 1e3}
    sim_dt = np.diff(arrays["time"])
    if len(sim_dt):
        out["sim_period_ms_mean"] = float(sim_dt.mean() * 1e3)
        out["sim_period_max_error_ms"] = float(np.abs(sim_dt - period).max() * 1e3)
    wall = arrays["wall_dt"][1:]
    if len(wall):
        for q in (50, 90, 99):
            out[f"wall_step_ms_p{q}"] = float(np.percentile(wall, q) * 1e3)
        out["wall_step_ms_max"] = float(wall.max() * 1e3)
        out["wall_step_jitter_ms"] = float(
            (np.percentile(wall, 99) - np.percentile(wall, 50)) * 1e3
        )
        out["deadline_miss_ratio"] = float((wall > period).mean())
    return out


def wall_histogram(arrays: dict[str, np.ndarray], bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Counts and bin edges (ms) of per-step wall time, for the report."""
    wall_ms = arrays["wall_dt"][1:] * 1e3
    if not len(wall_ms):
        return np.zeros(0, dtype=int), np.zeros(1)
    counts, edges = np.histogram(wall_ms, bins=bins)
    return counts, edges
