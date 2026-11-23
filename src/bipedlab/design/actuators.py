"""Actuator candidates and their fit to the WP4-01 joint requirements (WP4-03).

Each candidate is one servo at one supply voltage, with the manufacturer's stall
torque and no-load speed. Retail prices are a snapshot (date and source given) and
only orient the choice: nothing here is a purchase decision.

The fit uses the same 2x convention as WP4-01: stall torque >= 2 x the estimated
walking peak torque, and no-load speed >= 2 x the estimated walking peak speed. It
also reports the speed a DC motor still reaches at the walking peak torque
(``no_load * (1 - peak / stall)``, linear torque-speed line), which must exceed the
walking peak speed itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

RPM = 2 * math.pi / 60
MARGIN = 2.0


@dataclass(frozen=True)
class Candidate:
    name: str
    volts: float
    stall_torque: float  # N*m
    no_load_speed: float  # rad/s
    stall_current: float  # A
    mass: float  # kg
    supply_range: tuple[float, float]  # V, manufacturer's input range
    price: str  # retail snapshot, informative only
    source: str  # where the electrical numbers come from


# Manufacturer data read on 2026-10-26 (cloud web research). Prices are single-unit
# retail, taxes and shipping vary by country.
CANDIDATES = [
    Candidate("XL330-M288-T", 5.0, 0.52, 103 * RPM, 1.47, 0.018, (3.7, 6.0),
              "EUR 40.20 incl. VAT (generationrobots.com)",
              "emanual.robotis.com/docs/en/dxl/x/xl330-m288/"),
    Candidate("XL330-M288-T", 6.0, 0.60, 123 * RPM, 1.74, 0.018, (3.7, 6.0),
              "EUR 40.20 incl. VAT (generationrobots.com)",
              "emanual.robotis.com/docs/en/dxl/x/xl330-m288/"),
    Candidate("XC330-M181-T", 5.0, 0.60, 129 * RPM, 1.80, 0.023, (3.7, 6.0),
              "not checked",
              "emanual.robotis.com/docs/en/dxl/x/xc330-m181/"),
    Candidate("XC330-M288-T", 5.0, 0.93, 81 * RPM, 1.80, 0.023, (3.7, 6.0),
              "USD 103.39 (robotshop.com, incl. US tariffs)",
              "emanual.robotis.com/docs/en/dxl/x/xc330-m288/"),
    Candidate("XC330-M288-T", 6.0, 1.10, 97 * RPM, 2.15, 0.023, (3.7, 6.0),
              "USD 103.39 (robotshop.com, incl. US tariffs)",
              "emanual.robotis.com/docs/en/dxl/x/xc330-m288/"),
    Candidate("XC330-T288-T", 11.1, 0.92, 65 * RPM, 0.80, 0.023, (6.5, 12.0),
              "EUR 110.40 incl. VAT (generationrobots.com)",
              "emanual.robotis.com/docs/en/dxl/x/xc330-t288/"),
    Candidate("XL430-W250-T", 11.1, 1.4, 57 * RPM, 1.3, 0.0572, (6.5, 12.0),
              "not checked in USD/EUR",
              "emanual.robotis.com/docs/en/dxl/x/xl430-w250/"),
    Candidate("STS3215 (7.4 V)", 7.4, 19.5 * 0.0981, (math.pi / 3) / 0.192, 2.5, 0.055,
              (4.0, 7.4), "USD 33.97 (robotshop.com, incl. US tariffs)",
              "Feetech STS3215 datasheet (translated, core-electronics.com.au)"),
    # Low-budget PWM hobby servos (G4.5, 2026-11-23, owner's budget): no position
    # feedback. Numbers are the common seller datasheets at 6 V, not checked against a
    # manufacturer page; stall currents are seller figures too. Prices: robotsepeti.com
    # showed these in stock but its prices were not readable from the cloud.
    Candidate("DS3218MG (6 V)", 6.0, 20.0 * 0.0981, (math.pi / 3) / 0.15, 2.5, 0.060,
              (4.8, 6.8), "robotsepeti.com, in stock; price not read (est. ~600 TL)",
              "DSServo DS3218MG seller datasheet (5 V 19 kg*cm 0.16 s; 6.8 V 21.5, 0.14)"),
    Candidate("MG996R (6 V)", 6.0, 11.0 * 0.0981, (math.pi / 3) / 0.14, 2.5, 0.055,
              (4.8, 7.2), "robotsepeti.com, in stock; price not read (est. ~300 TL)",
              "Tower Pro MG996R seller datasheet"),
    Candidate("MG90S (6 V)", 6.0, 2.2 * 0.0981, (math.pi / 3) / 0.08, 0.7, 0.0134,
              (4.8, 6.0), "robotistan.com 190.65 TL incl. VAT (sold out 2026-11-23)",
              "Tower Pro MG90S seller datasheet"),
]  # fmt: skip


@dataclass(frozen=True)
class Requirement:
    joint: str
    peak_torque: float  # N*m, estimated walking peak (before margin)
    peak_speed: float  # rad/s, estimated walking peak (before margin)
    rms_torque: float  # N*m


# WP4-01 estimates for biped_v1 (gha-linux, PR #23 CI;
# docs/experiments/2026-10-26-wp4-01-sizing.md).
WP4_01 = [
    Requirement("hip_yaw", 0.106, 1.121, 0.048),
    Requirement("hip_roll", 0.174, 1.274, 0.092),
    Requirement("hip_pitch", 0.182, 3.694, 0.091),
    Requirement("knee", 0.339, 4.980, 0.194),
    Requirement("ankle", 0.218, 2.630, 0.103),
]

# From our own learned gait (WP4-27; mac-m5, MAC-36, docs/experiments/mac36/): the
# MAC-25 policy and its 1.5 N*m fine-tune, scored with the force limit at 1.5 N*m so
# no joint is clipped. Peak torque = worst walking max, peak speed = worst walking p95
# |joint speed| and RMS = worst walking RMS, over both policies and DR off / on.
GAIT_MAC36 = [
    Requirement("hip_yaw", 0.200, 1.65, 0.055),
    Requirement("hip_roll", 0.680, 2.75, 0.187),
    Requirement("hip_pitch", 0.585, 7.10, 0.172),
    Requirement("knee", 0.475, 6.85, 0.113),
    Requirement("ankle", 0.410, 4.00, 0.124),
]
REQUIREMENTS = {"wp4-01": WP4_01, "mac36": GAIT_MAC36}


def fit(c: Candidate, r: Requirement) -> dict[str, float | bool]:
    loaded = c.no_load_speed * max(0.0, 1 - r.peak_torque / c.stall_torque)
    torque_margin = c.stall_torque / r.peak_torque
    speed_margin = c.no_load_speed / r.peak_speed
    return {
        "torque_margin": torque_margin,
        "speed_margin": speed_margin,
        "loaded_speed_rad_s": loaded,
        "loaded_speed_ok": loaded >= r.peak_speed,
        "rms_over_stall": r.rms_torque / c.stall_torque,
        "passes": torque_margin >= MARGIN and speed_margin >= MARGIN and loaded >= r.peak_speed,
    }


def motor_force_range(
    stall: np.ndarray, joint_speed: np.ndarray, no_load_speed: float
) -> tuple[np.ndarray, np.ndarray]:
    """DC-motor force range (lower, upper) at the given joint speeds (WP4-28).

    Driving along the motion is capped at stall * (1 - |w| / w0), down to 0 at the
    no-load speed w0; braking against the motion keeps the full stall limit.
    """
    w = joint_speed / no_load_speed
    drive = stall * np.clip(1.0 - np.abs(w), 0.0, 1.0)
    return -np.where(w < 0, drive, stall), np.where(w > 0, drive, stall)
