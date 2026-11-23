"""Sim-to-real comparison for Phase 5 stage 7 (WP5-05).

A hardware run of ``biped-runtime --device ... --log REAL.bplog`` records what
the robot knew each period: joint angles and speeds, IMU, the gated command,
the targets it sent and the supervisor's bits. :func:`replay_commands` turns the
logged command into ``--command`` segments, so the same policy can be run in the
sim bridge with the same command history; :func:`compare` puts the metrics of
both logs side by side.

Only what the robot measures is compared: yaw rate (gyro), tilt, joint tracking
(next-period position minus the target sent), target spread, the estimated
servo torque demand ``kp * (target - q)``, faults and supervisor events. Body
speed is not measured on the robot (no mocap / odometry yet), so the speed
comparison needs an external measurement (e.g. distance over time on a tape).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from bipedlab.telemetry import runtime_log

FAULT_MASK = 0xFF
NOT_COMPARED = ["body speed (not measured on the robot)", "foot contact", "trunk position"]


def command_segments(
    time: np.ndarray, command: np.ndarray, tol: float = 0.01
) -> list[tuple[float, float, float, float, float]]:
    """Piecewise-constant ``(start, end, vx, vy, wz)`` within ``tol`` of the log.

    Zero-command stretches are left out (the gate's default is zero). The end of
    the last segment is one period past the last record.
    """
    if len(time) == 0:
        return []

    def nonzero(c: np.ndarray) -> bool:
        return bool(np.abs(c).max() > 1e-9)

    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.02
    segs: list[tuple[float, float, float, float, float]] = []
    start, ref = 0, command[0]
    for k in range(1, len(time) + 1):
        if (
            k == len(time)
            or np.abs(command[k] - ref).max() > tol
            or nonzero(command[k]) != nonzero(ref)
        ):
            end = time[k] if k < len(time) else time[-1] + dt
            mean = command[start:k].mean(axis=0)
            if nonzero(ref):
                vx, vy, wz = (float(v) for v in mean)
                segs.append((float(time[start]), float(end), vx, vy, wz))
            if k < len(time):
                start, ref = k, command[k]
    return segs


def runtime_command_args(segs: list[tuple[float, float, float, float, float]]) -> list[str]:
    out: list[str] = []
    for s in segs:
        out += ["--command", ":".join(f"{v:.6g}" for v in s)]
    return out


def metrics(arrays: dict[str, np.ndarray], header: dict[str, Any]) -> dict[str, Any]:
    """What the robot itself can tell about one run."""
    n_periods = len(arrays["time"])
    if n_periods < 2:
        raise ValueError("need at least two records")
    on = arrays["torque_enable"].astype(bool)
    q, target = arrays["qpos"], arrays["targets"]
    # A target sent in period k is reached (or not) by the state of period k + 1.
    both_on = on[:-1] & on[1:]
    track = (q[1:] - target[:-1])[both_on]
    torque = np.clip(header["kp"] * (target - q), -header["torque_limit"], header["torque_limit"])
    torque = np.abs(torque[on])
    x, y = arrays["quat"][:, 1], arrays["quat"][:, 2]
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)))
    moving = np.abs(arrays["command"]).max(axis=1) > 1e-6
    bits = arrays["bits"]
    latched = int(np.bitwise_or.reduce(bits & FAULT_MASK)) if len(bits) else 0
    events = {k: v for k, v in runtime_log.bit_counts(bits).items() if v}

    def per_joint(v: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> list[float] | None:
        return [float(x) for x in f(v)] if len(v) else None

    return {
        "periods": n_periods,
        "duration_s": float(arrays["time"][-1] - arrays["time"][0]),
        "torque_on_fraction": float(on.mean()),
        "latched_faults": [
            name for b, name in runtime_log.FAULT_NAMES.items() if b <= FAULT_MASK and latched & b
        ],  # fmt: skip
        "events": events,
        "yaw_rate_mean": float(arrays["gyro"][moving, 2].mean()) if moving.any() else None,
        "tilt_max_deg": float(tilt.max()),
        "tracking_rms": per_joint(track, lambda v: np.sqrt((v**2).mean(axis=0))),
        "target_std": per_joint(target[on], lambda v: v.std(axis=0)),
        "torque_p95": per_joint(torque, lambda v: np.percentile(v, 95, axis=0)),
        "bus_voltage_min": float(arrays["bus_voltage"].min()),
        "temperature_max": float(arrays["max_temperature"].max()),
    }


def compare(real: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    """Side by side, with ``real - sim`` for the numbers and per-joint ratios."""

    def diff(a: Any, b: Any) -> Any:
        if a is None or b is None:
            return None
        if isinstance(a, list):
            return [x - y for x, y in zip(a, b, strict=True)]
        return a - b

    def ratio(a: list[float] | None, b: list[float] | None) -> list[float | None] | None:
        if a is None or b is None:
            return None
        return [x / y if y > 1e-9 else None for x, y in zip(a, b, strict=True)]

    numeric = ["duration_s", "torque_on_fraction", "yaw_rate_mean", "tilt_max_deg",
               "bus_voltage_min", "temperature_max"]  # fmt: skip
    return {
        "real": real,
        "sim": sim,
        "real_minus_sim": {k: diff(real[k], sim[k]) for k in numeric},
        "tracking_ratio": ratio(real["tracking_rms"], sim["tracking_rms"]),
        "torque_p95_ratio": ratio(real["torque_p95"], sim["torque_p95"]),
        "faults_only_real": sorted(set(real["latched_faults"]) - set(sim["latched_faults"])),
        "not_compared": [
            "body speed (not measured on the robot)",
            "foot contact",
            "trunk position",
        ],
    }
