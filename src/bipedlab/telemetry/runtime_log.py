"""Read the Rust runtime's telemetry log (WP4-13) and convert it to the WP2-01 schema.

``biped-runtime --log FILE`` writes one record per control period (format in
``runtime/src/telemetry.rs``). :func:`read` returns it as arrays; :func:`convert`
writes the usual ``.npz`` + JSON sidecar, so ``bipedlab report`` and
``bipedlab profile`` work on a runtime run.

The log holds only what the robot itself knows. Trunk position, body linear
velocity and foot contact are not measured on the robot, so they are left out
(the report skips them). Joint torque is not measured either: the converter
estimates it as the position servo's demand ``kp * (target - q)`` clipped to the
torque limit (zero with torque off), and the sidecar marks it as estimated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab import runtime_bridge as rb
from bipedlab.telemetry import recorder as rec

MAGIC = b"BPLG"
VERSION = 1

FAULT_NAMES = {
    rb.FAULT_TILT: "tilt",
    rb.FAULT_OVERSPEED: "overspeed",
    rb.FAULT_NONFINITE_STATE: "nonfinite_state",
    rb.FAULT_STATE_TIMEOUT: "state_timeout",
    rb.FAULT_ESTOP: "estop",
    rb.FAULT_UNDERVOLTAGE: "undervoltage",
    rb.FAULT_OVERTEMPERATURE: "overtemperature",
    rb.EVENT_NONFINITE_TARGET: "nonfinite_target",
    rb.EVENT_POSITION_CLAMP: "position_clamp",
    rb.EVENT_SLEW_LIMIT: "slew_limit",
    rb.EVENT_TORQUE_CLAMP: "torque_clamp",
    rb.EVENT_DERATE: "derate",
}


def header_size(n: int) -> int:
    return 4 + 2 + 2 + 8 + 16 * n


def record_dtype(n: int) -> np.dtype:
    return np.dtype(
        [
            ("step", "<u8"),
            ("time", "<f8"),
            ("wall", "<f8"),
            ("compute_us", "<u4"),
            ("quat", "<f8", (4,)),
            ("gyro", "<f8", (3,)),
            ("bus_voltage", "<f8"),
            ("max_temperature", "<f8"),
            ("estop", "u1"),
            ("qpos", "<f8", (n,)),
            ("qvel", "<f8", (n,)),
            ("command", "<f8", (3,)),
            ("wanted", "<f8", (n,)),
            ("targets", "<f8", (n,)),
            ("torque_enable", "u1"),
            ("bits", "<u4"),
        ]
    )


def read(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Records as arrays (one row per period) and the header (dt, kp, torque_limit)."""
    raw = Path(path).read_bytes()
    if len(raw) < 8 or raw[:4] != MAGIC:
        raise ValueError("not a runtime log (bad magic)")
    version, n = np.frombuffer(raw, "<u2", 2, 4)
    if version != VERSION:
        raise ValueError(f"unsupported runtime log version {version}")
    n = int(n)
    if len(raw) < header_size(n):
        raise ValueError("truncated runtime log header")
    dt = float(np.frombuffer(raw, "<f8", 1, 8)[0])
    joints = np.frombuffer(raw, "<f8", 2 * n, 16).reshape(n, 2)
    dtype = record_dtype(n)
    body = raw[header_size(n) :]
    whole = len(body) // dtype.itemsize  # a cut-off last record (crash) is dropped
    recs = np.frombuffer(body, dtype, whole)
    arrays = {name: np.array(recs[name]) for name in dtype.names or ()}
    header = {
        "control_dt": dt,
        "n": n,
        "kp": joints[:, 0].copy(),
        "torque_limit": joints[:, 1].copy(),
        "dropped_bytes": len(body) - whole * dtype.itemsize,
    }
    return arrays, header


def bit_counts(bits: np.ndarray) -> dict[str, int]:
    """Periods in which each fault / supervisor event bit was set."""
    return {name: int(((bits & b) != 0).sum()) for b, name in FAULT_NAMES.items()}


EXTRA_CHANNELS: dict[str, tuple[str, str]] = {
    "joint_wanted": ("rad", "target the controller asked for, before the safety supervisor"),
    "torque_enable": ("bool", "servo torque on as sent by the supervisor"),
    "fault_bits": ("-", "latched faults (bits 0-7) and supervisor events (bits 8+)"),
    "compute_dt": ("s", "runtime compute time from state received to action sent"),
    "bus_voltage": ("V", "servo bus voltage"),
    "max_temperature": ("degC", "hottest servo"),
}


def to_telemetry(arrays: dict[str, np.ndarray], header: dict[str, Any]) -> dict[str, np.ndarray]:
    q, qd, target = arrays["qpos"], arrays["qvel"], arrays["targets"]
    on = arrays["torque_enable"].astype(bool)[:, None]
    limit = header["torque_limit"]
    torque = np.clip(header["kp"] * (target - q), -limit, limit) * on
    x, y = arrays["quat"][:, 1], arrays["quat"][:, 2]
    wall = arrays["wall"]
    return {
        "time": arrays["time"],
        "wall_dt": np.concatenate([[0.0], np.diff(wall)]) if len(wall) else wall,
        "joint_pos": q,
        "joint_vel": qd,
        "joint_target": target,
        "joint_torque": torque,
        "power": np.abs(torque * qd).sum(axis=1),
        "trunk_quat": arrays["quat"],
        "tilt": np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))),
        "gyro": arrays["gyro"],
        "command": arrays["command"],
        "joint_wanted": arrays["wanted"],
        "torque_enable": arrays["torque_enable"].astype(bool),
        "fault_bits": arrays["bits"],
        "compute_dt": arrays["compute_us"] * 1e-6,
        "bus_voltage": arrays["bus_voltage"],
        "max_temperature": arrays["max_temperature"],
    }


def convert(
    log_path: str | Path,
    out: str | Path,
    joint_names: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``out`` (.npz) and its JSON sidecar from a runtime log."""
    raw, header = read(log_path)
    if not len(raw["time"]):
        raise ValueError("runtime log has no records")
    arrays = to_telemetry(raw, header)
    out = Path(out).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)  # type: ignore[arg-type]
    channels = {**rec.CHANNELS, **EXTRA_CHANNELS}
    sidecar = {
        "schema_version": rec.SCHEMA_VERSION,
        "source": "biped-runtime log",
        "channels": {
            name: {
                "unit": channels[name][0],
                "description": channels[name][1],
                "shape": list(arrays[name].shape),
            }
            for name in arrays
        },
        "estimated": {"joint_torque": "kp * (target - q) clipped to the torque limit"},
        "not_measured": ["trunk_pos", "lin_vel_body", "foot_contact"],
        "joint_names": joint_names or [f"j{k}" for k in range(header["n"])],
        "meta": {
            "source": "biped-runtime --log",
            "control_dt_s": header["control_dt"],
            "periods": int(len(arrays["time"])),
            "supervisor_bits_periods": bit_counts(raw["bits"]),
            **(meta or {}),
        },
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    return out
