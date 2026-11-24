"""Bench calibration of the DS3218MG build's pulse mapping (WP5-18, stage 3).

The supervisor turns a joint target ``q`` (sim frame, rad) into a pulse with
``pulse = center + sign * us_per_rad * (q + offset)`` (``firmware/supervisor-core/src/
servo.rs``). On the bench the servo horn sits on a 25-tooth spline, the mirrored side turns
the other way and each servo's gain differs a little, so the real joint angle is not ``q``.

Procedure: command each joint, one at a time, to the angles of :func:`plan`, measure the
real joint angle (degrees, phone inclinometer or protractor, zero = the joint's sim zero)
and write one CSV row per reading: ``joint,commanded_rad,measured_deg``. :func:`fit` fits
``measured = gain * commanded + bias`` per joint and :func:`corrected` folds it into a new
``JointPwm`` so that the real angle equals the target. :func:`check` refuses a fit that
points at a wrong horn tooth, a wrong servo or a bad reading, and :func:`rust_snippet`
prints the result for ``joints.rs``.

Nothing here writes to the firmware: the mapping is a supervisor limit and changes there
only with the owner's approval (ADR-0015).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab import REPO_ROOT

JOINTS_RS = REPO_ROOT / "firmware/supervisor-core/src/joints.rs"
LEG = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
JOINT_NAMES = tuple(f"{side}_{j}" for side in ("left", "right") for j in LEG)

US_PER_RAD = 2000.0 / math.pi  # DS3218MG 180 deg: 500-2500 us over pi rad
CENTER_US = 1500.0
MIN_US, MAX_US = 500, 2500

# Acceptance limits for one joint's fit.
GAIN_RANGE = (0.85, 1.15)  # |gain|; outside: wrong servo model or a bad reading
MAX_BIAS = math.radians(7.2)  # half a spline tooth (360/25/2 deg); more: move the horn
MAX_RMS = math.radians(2.0)  # fit residual; more: slack in the bracket or a misread
MIN_POINTS = 3
# Range the servo cannot reach after trim: the supervisor clamps the pulse there and the
# joint stops short. Hip pitch already loses 0.03 rad (joints.rs); refuse beyond 5 deg.
MAX_UNREACHABLE = math.radians(5.0)


@dataclass(frozen=True)
class JointPwm:
    """Python twin of the firmware's ``JointPwm``."""

    center_us: float
    us_per_rad: float
    sign: float
    offset: float
    lower: float
    upper: float
    min_us: int = MIN_US
    max_us: int = MAX_US

    def pulse_us(self, q: float) -> int:
        us = self.center_us + self.sign * self.us_per_rad * (q + self.offset)
        if not math.isfinite(us):
            us = self.center_us
        return int(min(max(int(us + 0.5), self.min_us), self.max_us))

    def angle(self, pulse: float) -> float:
        """Joint angle this mapping assumes for a pulse (inverse of ``pulse_us``)."""
        return (pulse - self.center_us) / (self.sign * self.us_per_rad) - self.offset


def firmware_joints(path: Path = JOINTS_RS) -> list[JointPwm]:
    """The firmware's nominal mapping, read from ``joints.rs`` (single source of truth)."""
    src = path.read_text()
    block = re.search(r"pub const RANGES[^=]*=\s*\[(.*?)\];", src, re.S)
    knee = re.search(r"pub const KNEE_CENTER: f32 = ([-\d.]+);", src)
    if block is None or knee is None:
        raise ValueError(f"RANGES / KNEE_CENTER not found in {path}")
    ranges = re.findall(r"\(\s*([-\d.]+),\s*([-\d.]+)\s*\)", block.group(1))
    out = []
    for k, (lo, hi) in enumerate(ranges):
        offset = -float(knee.group(1)) if k % 5 == 3 else 0.0
        out.append(JointPwm(CENTER_US, US_PER_RAD, 1.0, offset, float(lo), float(hi)))
    if len(out) != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} joints, found {len(out)}")
    return out


def plan(joints: list[JointPwm] | None = None, points: int = 5) -> list[dict[str, Any]]:
    """Angles to command per joint: evenly inside the middle 60 % of its range.

    The middle of the range keeps the horn away from the end stops and a mounted leg away
    from its neighbours; rounding to 0.01 rad keeps the rows easy to type.
    """
    joints = joints or firmware_joints()
    rows = []
    for name, j in zip(JOINT_NAMES, joints, strict=True):
        span = j.upper - j.lower
        for x in np.linspace(j.lower + 0.2 * span, j.upper - 0.2 * span, points):
            q = round(float(x), 2)
            rows.append({"joint": name, "commanded_rad": q, "pulse_us": j.pulse_us(q)})
    return rows


def read_measurements(path: Path) -> dict[str, list[tuple[float, float]]]:
    """CSV ``joint,commanded_rad,measured_deg`` -> {joint: [(commanded, measured rad)]}."""
    out: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["joint"].strip()
            if name not in JOINT_NAMES:
                raise ValueError(f"unknown joint {name!r}")
            q = float(row["commanded_rad"])
            out.setdefault(name, []).append((q, math.radians(float(row["measured_deg"]))))
    return out


@dataclass(frozen=True)
class Fit:
    gain: float
    bias: float  # rad
    rms: float  # rad
    n: int


def fit(samples: list[tuple[float, float]]) -> Fit:
    """Least squares ``measured = gain * commanded + bias``."""
    q = np.array([s[0] for s in samples])
    m = np.array([s[1] for s in samples])
    if len(q) < 2 or np.ptp(q) < 1e-6:
        raise ValueError("need readings at two or more different angles")
    a = np.vstack([q, np.ones_like(q)]).T
    (gain, bias), *_ = np.linalg.lstsq(a, m, rcond=None)
    rms = float(np.sqrt(np.mean((a @ np.array([gain, bias]) - m) ** 2)))
    return Fit(float(gain), float(bias), rms, len(q))


def corrected(j: JointPwm, f: Fit) -> JointPwm:
    """New mapping whose real angle equals the target.

    The servo does ``measured = gain * q + bias`` for a command ``q``, so commanding
    ``(q - bias) / gain`` lands on ``q``. Folded into the mapping that is
    ``sign' = sign * sgn(gain)``, ``us_per_rad' = us_per_rad / |gain|`` and
    ``offset' = gain * offset - bias``.
    """
    return replace(
        j,
        sign=j.sign * math.copysign(1.0, f.gain),
        us_per_rad=j.us_per_rad / abs(f.gain),
        offset=f.gain * j.offset - f.bias,
    )


def check(name: str, j: JointPwm, f: Fit) -> list[str]:
    """Reasons to refuse this joint's calibration (empty: accept)."""
    problems = []
    if f.n < MIN_POINTS:
        problems.append(f"{name}: {f.n} readings, need {MIN_POINTS}")
    if not GAIN_RANGE[0] <= abs(f.gain) <= GAIN_RANGE[1]:
        problems.append(f"{name}: gain {f.gain:.3f} outside ±{GAIN_RANGE} (wrong servo?)")
    if abs(f.bias) > MAX_BIAS:
        problems.append(
            f"{name}: bias {math.degrees(f.bias):.1f} deg > half a spline tooth: "
            "move the horn one tooth and measure again"
        )
    if f.rms > MAX_RMS:
        problems.append(f"{name}: fit residual {math.degrees(f.rms):.1f} deg (slack or misread)")
    lost = unreachable(corrected(j, f))
    if lost > MAX_UNREACHABLE:
        problems.append(
            f"{name}: {math.degrees(lost):.1f} deg of the range is past the servo travel"
        )
    return problems


def unreachable(j: JointPwm) -> float:
    """Largest part of the joint range (rad) the servo travel cannot reach."""
    worst = 0.0
    for q in (j.lower, j.upper):
        us = j.center_us + j.sign * j.us_per_rad * (q + j.offset)
        over = max(j.min_us - us, us - j.max_us, 0.0)
        worst = max(worst, over / j.us_per_rad)
    return worst


def calibrate(measurements: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
    """Fit every measured joint; untouched joints keep the nominal mapping."""
    joints = firmware_joints()
    report: dict[str, Any] = {"joints": {}, "problems": [], "missing": []}
    for name, j in zip(JOINT_NAMES, joints, strict=True):
        if name not in measurements:
            report["missing"].append(name)
            continue
        f = fit(measurements[name])
        problems = check(name, j, f)
        report["problems"] += problems
        report["joints"][name] = {
            "fit": {
                "gain": round(f.gain, 4),
                "bias_deg": round(math.degrees(f.bias), 2),
                "rms_deg": round(math.degrees(f.rms), 2),
                "n": f.n,
            },
            "pwm": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(corrected(j, f)).items()
            },
            "unreachable_deg": round(math.degrees(unreachable(corrected(j, f))), 1),
            "ok": not problems,
        }
    report["ok"] = not report["problems"] and not report["missing"]
    return report


def rust_snippet(report: dict[str, Any]) -> str:
    """``joints.rs`` body for the calibrated mapping, for review (not applied)."""
    lines = [
        "// Bench calibration (WP5-18). Review and apply only with the owner's",
        "// approval (ADR-0015).",
        "pub fn biped_ds3218_calibrated() -> [JointPwm; N_JOINTS] {",
        "    let mut j = biped_ds3218();",
    ]
    for k, name in enumerate(JOINT_NAMES):
        row = report["joints"].get(name)
        if row is None:
            lines.append(f"    // {name}: not measured, nominal")
            continue
        p = row["pwm"]
        lines.append(
            f"    j[{k}].sign = {p['sign']:.1f}; j[{k}].us_per_rad = {p['us_per_rad']:.2f}; "
            f"j[{k}].offset = {p['offset']:.4f}; // {name}"
        )
    lines += ["    j", "}"]
    return "\n".join(lines)


def synthetic_measurements(
    true_gain: dict[str, float], true_bias: dict[str, float], noise_deg: float = 0.5,
    seed: int = 0,
) -> dict[str, list[tuple[float, float]]]:  # fmt: skip
    """Readings a servo with this gain / bias would give for :func:`plan` (for tests)."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[float, float]]] = {}
    for row in plan():
        name, q = row["joint"], row["commanded_rad"]
        m = true_gain.get(name, 1.0) * q + true_bias.get(name, 0.0)
        m += math.radians(float(rng.normal(0.0, noise_deg)))
        out.setdefault(name, []).append((q, m))
    return out
