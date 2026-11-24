# ruff: noqa: E501  (MJCF template lines)
"""Parametric MJCF for our own biped (WP4-02).

``BipedParams`` holds every dimension and mass of the design; ``to_mjcf`` turns it
into a MuJoCo model built only from primitive geometry (boxes), so the result owns
no third-party geometry (ADR-0004). The leg layout is the common small-biped one:
per leg hip yaw, hip roll, hip pitch, knee and ankle pitch (10 joints).

Masses are lumped where the parts sit: each servo is placed on the parent side of
the joint it drives (hip-yaw servos in the trunk, knee servo at the bottom of the
thigh, and so on), links and brackets carry their own mass, and the trunk carries
the electronics and the battery. The numbers are a budget for WP4-01, not a
measured robot.

Frame: x forward, y left, z up. Pitch joints turn about +y, so a positive hip-pitch,
knee or ankle angle swings the part below it backwards; a bent knee is positive.
``crouch_pose`` gives the angles that keep the trunk upright and the feet flat.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
SIDES = ("left", "right")


@dataclass(frozen=True)
class ServoSpec:
    """Placeholder actuator class; WP4-03 picks the real part."""

    name: str = "xl330-class"
    mass: float = 0.018  # kg
    size: tuple[float, float, float] = (0.020, 0.034, 0.026)  # x, y, z (m)
    torque_limit: float = 0.52  # N*m, stall at the supply voltage (sizing input)
    speed_limit: float = 10.0  # rad/s, no-load
    kp: float = 5.0  # N*m/rad, position-servo stiffness in sim
    kv: float = 0.1  # N*m*s/rad
    armature: float = 0.002  # kg*m^2, reflected rotor inertia
    damping: float = 0.05
    frictionloss: float = 0.01


@dataclass(frozen=True)
class BipedParams:
    name: str = "biped_v1"
    servo: ServoSpec = field(default_factory=ServoSpec)
    # Trunk.
    trunk_size: tuple[float, float, float] = (0.070, 0.100, 0.075)  # full extents (m)
    trunk_structure_mass: float = 0.110  # frame and covers
    electronics_mass: float = 0.040  # computer, IMU, servo bus board, wiring
    battery_mass: float = 0.050  # 2S LiPo, ~500 mAh
    battery_offset: tuple[float, float, float] = (0.0, 0.0, -0.015)  # from trunk centre
    # Legs (lengths along the leg at zero angles).
    hip_spacing: float = 0.080  # between the hip-yaw axes (m)
    trunk_to_hip: float = 0.030  # trunk centre down to the hip-yaw joint
    yaw_to_roll: float = 0.015
    roll_to_pitch: float = 0.015
    thigh: float = 0.065
    shin: float = 0.065
    ankle_height: float = 0.026  # ankle axis to sole
    link_width: float = 0.022  # bracket/link cross-section, x and y
    bracket_mass: float = 0.008  # each hip yaw/roll bracket
    thigh_mass: float = 0.015
    shin_mass: float = 0.015
    foot_size: tuple[float, float, float] = (0.090, 0.050, 0.006)  # full extents
    foot_forward: float = 0.010  # sole centre ahead of the ankle axis
    foot_mass: float = 0.020
    foot_friction: float = 1.0
    # Joint ranges (rad).
    ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "hip_yaw": (-0.5, 0.5),
            "hip_roll": (-0.4, 0.4),
            "hip_pitch": (-1.6, 1.6),
            "knee": (-0.1, 2.2),
            "ankle_pitch": (-1.2, 1.2),
        }
    )
    stand_knee: float = 0.6  # knee bend of the STAND keyframe (rad)
    timestep: float = 0.005
    # Servo boxes: "lumped" centres each servo on the joint it drives, as ``servo.size``
    # (a mass budget, not a build). "u_bracket" places them as a U-bracket build holds
    # them (WP5-13): the servo is centred on its joint along the axis (the bracket wraps
    # both faces), its height (``size[2]``) lies along the joint axis, its length
    # (``size[1]``) along the leg, with the shaft ``shaft_from_end`` from the end nearest
    # the joint; hip-yaw servos stand in the trunk with ``bracket_gap`` below them.
    servo_layout: str = "lumped"
    shaft_from_end: float = 0.010
    bracket_gap: float = 0.005
    # Servo boxes collide with the other leg's servos and feet (not the floor), so a
    # policy cannot cross its legs through each other (WP5-13).
    leg_collisions: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Low-budget variant (G4.5, 2026-11-23): the owner picked ten DSServo DS3218MG PWM hobby
# servos (sold in Turkey, ~1/3 of the bus-servo cost). Electrical numbers are the
# seller's datasheet (5 V: 19 kg*cm, 0.16 s/60 deg; 6.8 V: 21.5 kg*cm, 0.14 s/60 deg),
# taken at the 6 V supply by linear interpolation: 20 kg*cm = 1.96 N*m and
# 0.15 s/60 deg = 7.0 rad/s. 60 g, 40 x 20 x 40.5 mm. The position-loop gains, reflected
# inertia and friction are guesses for a geared hobby servo until a bench
# identification (PHASE5_BRINGUP_PLAN.md) measures them; DR should cover them.
DS3218_SERVO = ServoSpec(
    name="ds3218mg",
    mass=0.060,
    size=(0.020, 0.040, 0.0405),
    torque_limit=1.96,
    speed_limit=7.0,
    kp=10.0,
    kv=0.2,
    armature=0.004,
    damping=0.1,
    frictionloss=0.03,
)

# The same kinematics as biped_v1 (legs, ranges, stand pose), so the gait work carries
# over, with the heavier parts the DS3218MG build needs: a larger trunk for two
# 40 mm hip-yaw servos, a 2S 1200 mAh pack (~70 g), the PWM electronics (Pi Zero 2 W,
# RP2040 or PCA9685, MPU6050, 6 V UBEC, wiring ~50 g) and stiffer brackets and links.
BIPED_DS3218 = BipedParams(
    name="biped_ds3218",
    servo=DS3218_SERVO,
    trunk_size=(0.080, 0.110, 0.080),
    trunk_structure_mass=0.140,
    electronics_mass=0.050,
    battery_mass=0.070,
    bracket_mass=0.012,
    thigh_mass=0.020,
    shin_mass=0.020,
    foot_mass=0.025,
)

# Buildable DS3218MG layout (WP5-13, 2026-11-24). biped_ds3218 keeps the 18 g servo
# kinematics, where three 40 mm servos share 30 mm of hip: the hip-roll and hip-pitch
# servo boxes overlap by 20 mm, so it cannot be built. Here the servos sit as U-brackets
# hold them (``servo_layout="u_bracket"``) and the hip grows until every pair of servo
# boxes clears by at least 3 mm in the stand pose and at each joint's range ends
# when only one leg moves (``scripts/check_clearance.py``). Same ranges, legs and stand pose; the hip pitch axis is
# 50 mm lower and the legs collide with each other, so the policy must be retrained
# (MAC-50).
BIPED_DS3218B = replace(
    BIPED_DS3218,
    name="biped_ds3218b",
    servo_layout="u_bracket",
    yaw_to_roll=0.030,
    roll_to_pitch=0.050,
    leg_collisions=True,
    trunk_size=(0.080, 0.110, 0.070),  # 10 mm flatter: stays under G4.2's 30 cm
)

# Mass check for the off-the-shelf frame (WP5-15): MG995/MG996 aluminium brackets weigh
# ~25-30 g per joint (hardware/bom/frame_ds3218b.md), more than biped_ds3218b budgets.
# Same geometry with heavier brackets, links and feet: ~1.17 kg instead of 1.04 kg.
# Only for evaluating / fine-tuning against the heavier build until real parts are weighed.
BIPED_DS3218B_HEAVY = replace(
    BIPED_DS3218B,
    name="biped_ds3218b_heavy",
    bracket_mass=0.025,
    thigh_mass=0.035,
    shin_mass=0.035,
    foot_mass=0.035,
)

VARIANTS = {
    "biped_v1": BipedParams(),
    "biped_ds3218": BIPED_DS3218,
    "biped_ds3218b": BIPED_DS3218B,
    "biped_ds3218b_heavy": BIPED_DS3218B_HEAVY,
}


def crouch_pose(knee: float) -> dict[str, float]:
    """Joint angles that keep trunk upright and feet flat for equal thigh and shin.

    Bending the knee by ``knee`` tilts the shin back by ``knee`` relative to the
    thigh; hip pitch and ankle each take half of it with opposite signs.
    """
    return {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": -knee / 2, "knee": knee,
            "ankle_pitch": -knee / 2}  # fmt: skip


def stand_height(p: BipedParams, knee: float) -> float:
    """Height of the trunk centre above the floor in the crouch pose."""
    leg = (p.thigh + p.shin) * np.cos(knee / 2)
    return float(p.trunk_to_hip + p.yaw_to_roll + p.roll_to_pitch + leg + p.ankle_height)


def _v(x: Sequence[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.6g}" for v in x)


def _box(name: str, size: tuple[float, float, float], pos: tuple[float, ...], mass: float,
         cls: str | None = None) -> str:  # fmt: skip
    half = tuple(s / 2 for s in size)
    c = f' class="{cls}"' if cls else ""
    return (
        f'<geom name="{name}" type="box" size="{_v(half)}" pos="{_v(pos)}" mass="{mass:.6g}"{c}/>'
    )


def servo_box(
    p: BipedParams, axis: str, joint_z: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Full size and centre of a leg servo box for a joint on axis ``x`` or ``y``."""
    sv = p.servo
    if p.servo_layout == "lumped":
        return sv.size, (0.0, 0.0, joint_z)
    if p.servo_layout != "u_bracket":
        raise ValueError(f"unknown servo_layout {p.servo_layout!r}")
    w, length, h = sv.size
    size = (h, w, length) if axis == "x" else (w, h, length)
    # Body above the shaft (towards the parent): centre length/2 - shaft_from_end up.
    return size, (0.0, 0.0, joint_z + length / 2 - p.shaft_from_end)


def yaw_servo_box(
    p: BipedParams, side: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Full size and centre of a hip-yaw servo box in the trunk frame."""
    sv = p.servo
    y = (1.0 if side == "left" else -1.0) * p.hip_spacing / 2
    if p.servo_layout == "lumped":
        return sv.size, (0.0, y, -p.trunk_to_hip + sv.size[2] / 2)
    w, length, h = sv.size
    # Length along x so the two yaw servos sit side by side without touching.
    return (length, w, h), (0.0, y, -p.trunk_to_hip + p.bracket_gap + h / 2)


def _leg(p: BipedParams, side: str) -> str:
    s = 1.0 if side == "left" else -1.0
    sv, w = p.servo, p.link_width
    r = p.ranges

    def joint(j: str, axis: str) -> str:
        lo, hi = r[j]
        if side == "right" and j in ("hip_yaw", "hip_roll"):
            lo, hi = -hi, -lo  # mirror the asymmetric ranges
        return f'<joint name="{side}_{j}" axis="{axis}" range="{lo:.6g} {hi:.6g}"/>'

    def g(name: str) -> str:
        return f"{side}_{name}"

    def servo(name: str, axis: str, joint_z: float) -> str:
        """Servo box of the joint at ``joint_z`` (in the parent body's frame)."""
        size, pos = servo_box(p, axis, joint_z)
        return _box(g(name), size, pos, sv.mass, "servo")

    foot_z = -(p.ankle_height - p.foot_size[2] / 2)  # foot box centre; sole at -ankle_height
    return f"""
      <body name="{side}_hip_yaw_link" pos="0 {s * p.hip_spacing / 2:.6g} {-p.trunk_to_hip:.6g}">
        {joint("hip_yaw", "0 0 1")}
        {
        _box(
            g("hip_yaw_bracket"), (w, w, p.yaw_to_roll), (0, 0, -p.yaw_to_roll / 2), p.bracket_mass
        )
    }
        {servo("hip_roll_servo", "x", -p.yaw_to_roll)}
        <body name="{side}_hip_roll_link" pos="0 0 {-p.yaw_to_roll:.6g}">
          {joint("hip_roll", "1 0 0")}
          {
        _box(
            g("hip_roll_bracket"),
            (w, w, p.roll_to_pitch),
            (0, 0, -p.roll_to_pitch / 2),
            p.bracket_mass,
        )
    }
          {servo("hip_pitch_servo", "y", -p.roll_to_pitch)}
          <body name="{side}_thigh" pos="0 0 {-p.roll_to_pitch:.6g}">
            {joint("hip_pitch", "0 1 0")}
            {_box(g("thigh"), (w, w, p.thigh), (0, 0, -p.thigh / 2), p.thigh_mass)}
            {servo("knee_servo", "y", -p.thigh)}
            <body name="{side}_shin" pos="0 0 {-p.thigh:.6g}">
              {joint("knee", "0 1 0")}
              {_box(g("shin"), (w, w, p.shin), (0, 0, -p.shin / 2), p.shin_mass)}
              {servo("ankle_servo", "y", -p.shin)}
              <body name="{side}_foot" pos="0 0 {-p.shin:.6g}">
                {joint("ankle_pitch", "0 1 0")}
                {_box(g("foot"), p.foot_size, (p.foot_forward, 0, foot_z), p.foot_mass, "foot")}
                <site name="{side}_foot" pos="{p.foot_forward:.6g} 0 {-p.ankle_height:.6g}"/>
              </body>
            </body>
          </body>
        </body>
      </body>"""


def to_mjcf(p: BipedParams | None = None) -> str:
    """MJCF text of the design; deterministic for a given ``BipedParams``."""
    p = p or BipedParams()
    sv = p.servo
    tx, ty, tz = p.trunk_size
    z0 = stand_height(p, p.stand_knee) + 0.002
    pose = crouch_pose(p.stand_knee)
    leg_q = [pose[j] for j in LEG_JOINTS]
    # Right-leg yaw/roll are mirrored; the crouch has them at zero so order is enough.
    stand_q = leg_q * 2
    yaw_l, yaw_r = yaw_servo_box(p, "left"), yaw_servo_box(p, "right")
    servo_contact = ' contype="2" conaffinity="2"' if p.leg_collisions else ""
    foot_affinity = 3 if p.leg_collisions else 1
    actuators = "\n".join(
        f'    <position name="{side}_{j}" joint="{side}_{j}"/>'
        for side in SIDES
        for j in LEG_JOINTS
    )
    return f"""<!--
  {p.name}: our own biped, generated by bipedlab.design.biped (WP4-02).
  Do not edit by hand: change BipedParams and regenerate with
  `uv run python scripts/gen_biped.py`.
  Original primitive geometry only (ADR-0004). License: Apache-2.0 (this repository).
  Servo class: {sv.name}, {sv.torque_limit} N*m limit, {sv.mass * 1000:.0f} g.
-->
<mujoco model="{p.name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{p.timestep:.6g}" integrator="implicitfast"/>

  <default>
    <joint damping="{sv.damping:.6g}" armature="{sv.armature:.6g}" frictionloss="{sv.frictionloss:.6g}"/>
    <position kp="{sv.kp:.6g}" kv="{sv.kv:.6g}" forcerange="{-sv.torque_limit:.6g} {sv.torque_limit:.6g}"/>
    <geom contype="0" conaffinity="0" rgba="0.75 0.75 0.78 1"/>
    <default class="servo">
      <geom{servo_contact} rgba="0.15 0.15 0.15 1"/>
    </default>
    <default class="foot">
      <geom contype="1" conaffinity="{foot_affinity}" priority="1" friction="{p.foot_friction:.6g} 0.005 0.0001" rgba="0.2 0.3 0.6 1"/>
    </default>
  </default>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" contype="1" conaffinity="1" rgba="0.3 0.4 0.5 1"/>

    <body name="trunk" pos="0 0 {z0:.6g}">
      <freejoint name="root"/>
      {_box("trunk", (tx, ty, tz), (0, 0, 0), p.trunk_structure_mass)}
      {_box("electronics", (tx * 0.8, ty * 0.6, 0.01), (0, 0, tz / 2 - 0.01), p.electronics_mass)}
      {_box("battery", (0.05, 0.03, 0.015), p.battery_offset, p.battery_mass)}
      {_box("left_hip_yaw_servo", yaw_l[0], yaw_l[1], sv.mass, "servo")}
      {_box("right_hip_yaw_servo", yaw_r[0], yaw_r[1], sv.mass, "servo")}
      <site name="imu" pos="0 0 0"/>
{_leg(p, "left")}
{_leg(p, "right")}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>

  <sensor>
    <framequat name="orientation" objtype="site" objname="imu"/>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
  </sensor>

  <keyframe>
    <key name="STAND" qpos="0 0 {z0:.6g} 1 0 0 0  {_v(stand_q)}" ctrl="{_v(stand_q)}"/>
  </keyframe>
</mujoco>
"""


def write(path: str | Path, p: BipedParams | None = None) -> Path:
    path = Path(path)
    path.write_text(to_mjcf(p))
    return path
