"""WP4-01: sizing report for our biped (mass, CoM, joint torque/speed, power, battery).

Static loads are computed for our model (``BipedParams`` defaults) and for Microduck.
With ``--reference`` (a telemetry recording of Microduck's official walking policy,
``bipedlab record alpha_walking.onnx --actuator bam ...``) the walking peaks give a
dynamic factor per joint type, which turns our static torques into walking
estimates; requirements then carry the G4.1 margin (2x).

Usage:
  uv run python scripts/size_biped.py [--reference reports/telemetry/walk.npz] [--out f.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab import upstream
from bipedlab.design import biped, sizing
from bipedlab.sim import microduck_policy as mp
from bipedlab.telemetry import recorder

MARGIN = 2.0  # G4.1
CROUCH_KNEES = (0.6, 1.0, 1.4)
# Electrical assumptions (documented in the report; replace with datasheet values in WP4-03).
SERVO_EFFICIENCY = 0.25  # small DC gearmotor + gearbox, electrical -> mechanical
IDLE_W = 2.5  # computer + IMU + 10 servo controllers holding position
BATTERY_MAH, BATTERY_V = 500.0, 7.4  # 2S LiPo of the mass budget
RMS_FLAG = 0.3  # RMS torque above 30 % of stall: thermal check needed


def our_static(p: biped.BipedParams) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_string(biped.to_mjcf(p))
    out: dict[str, Any] = {"cases": []}
    for knee in CROUCH_KNEES:
        pose = biped.crouch_pose(knee)
        q = np.zeros(model.nq)
        q[2], q[3] = biped.stand_height(p, knee), 1.0
        for side in biped.SIDES:
            for j, v in pose.items():
                q[model.jnt_qposadr[model.joint(f"{side}_{j}").id]] = v
        for feet in (("left_foot", "right_foot"), ("left_foot",)):
            name = f"knee {knee:.1f}, {'double' if len(feet) == 2 else 'single'} support"
            out["cases"].append(sizing.static_loads(model, sizing.StaticCase(name, q, feet)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    out["com_height_m"] = float(data.subtree_com[1][2])
    out["com_x_from_ankle_m"] = float(data.subtree_com[1][0] - data.body("left_foot").xpos[0])
    out["leg_length_m"] = p.thigh + p.shin + p.ankle_height
    out["mass_kg"] = float(model.body_subtreemass[1])
    return out


def microduck_model() -> mujoco.MjModel:
    model, _ = mp.load_model_for_policy(upstream.microduck_scene("scene_walk.xml"), "mjcf")
    return model


def microduck_static(model: mujoco.MjModel) -> dict[str, Any]:
    q = np.zeros(model.nq)
    q[3] = 1.0
    q[model.jnt_qposadr[model.actuator_trnid[:, 0]]] = mp.DEFAULT_POSE
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    hip = data.body("upper_leg_left").xpos[2]
    return {
        "single": sizing.static_loads(model, sizing.StaticCase("HOME, single", q, ["left_foot"])),
        "mass_kg": float(model.body_subtreemass[1]),
        "leg_length_m": float(hip - data.site("left_foot").xpos[2]),
    }


def requirements(ours: dict[str, Any], duck: dict[str, Any], walk: dict[str, Any],
                 p: biped.BipedParams) -> dict[str, Any]:  # fmt: skip
    """Walking estimates for our robot and the required ratings with the margin."""
    ours_single = {}
    for c in ours["cases"]:
        if "single" in c["case"] and c["case"].startswith("knee 1.0"):
            ours_single = sizing.by_type(c["torque_nm"])
    duck_single = sizing.by_type(duck["single"]["torque_nm"])
    walk_peak = sizing.by_type({n: v["peak_torque_nm"] for n, v in walk["joints"].items()})
    walk_rms = sizing.by_type({n: v["rms_torque_nm"] for n, v in walk["joints"].items()})
    walk_speed = sizing.by_type({n: v["peak_speed_rad_s"] for n, v in walk["joints"].items()})
    # Geometric similarity: torque ~ m g L; speed ~ sqrt(g / L) (equal Froude number).
    scale_torque = ours["mass_kg"] * ours["leg_length_m"] / (duck["mass_kg"] * duck["leg_length_m"])
    scale_speed = math.sqrt(duck["leg_length_m"] / ours["leg_length_m"])
    scale_power = (
        ours["mass_kg"] / duck["mass_kg"] * math.sqrt(ours["leg_length_m"] / duck["leg_length_m"])
    )
    rows = {}
    for t in sizing.JOINT_TYPES:
        static_d, static_o = duck_single.get(t, 0.0), ours_single.get(t, 0.0)
        if static_d > 0.01 and static_o > 0.01:
            factor = walk_peak[t] / static_d
            peak = static_o * factor
            basis = "our static x Microduck walk/static"
        else:  # no static load (e.g. hip yaw): scale Microduck's walking peak
            factor = float("nan")
            peak = walk_peak[t] * scale_torque
            basis = "Microduck walking peak x m*L ratio"
        rms = walk_rms[t] * peak / walk_peak[t] if walk_peak[t] > 0 else 0.0
        speed = walk_speed[t] * scale_speed
        rows[t] = {
            "basis": basis,
            "dynamic_factor": factor,
            "walk_peak_torque_nm": peak,
            "walk_rms_torque_nm": rms,
            "walk_peak_speed_rad_s": speed,
            "required_torque_nm": MARGIN * peak,
            "required_speed_rad_s": MARGIN * speed,
            "torque_margin_vs_servo": p.servo.torque_limit / peak if peak > 0 else float("inf"),
            "speed_margin_vs_servo": p.servo.speed_limit / speed if speed > 0 else float("inf"),
            "rms_over_stall": rms / p.servo.torque_limit,
            "thermal_flag": rms / p.servo.torque_limit > RMS_FLAG,
        }
    mech_mean = walk["mech_power_mean_w"] * scale_power
    mech_peak = walk["mech_power_peak_w"] * scale_power
    elec = mech_mean / SERVO_EFFICIENCY + IDLE_W
    return {
        "scale_torque": scale_torque,
        "scale_speed": scale_speed,
        "joints": rows,
        "power": {
            "mech_mean_w": mech_mean,
            "mech_peak_w": mech_peak,
            "electrical_mean_w": elec,
            "electrical_peak_w": mech_peak / SERVO_EFFICIENCY + IDLE_W,
            "battery": f"{BATTERY_MAH:.0f} mAh {BATTERY_V} V",
            "runtime_min": sizing.battery_runtime_min(BATTERY_MAH, BATTERY_V, elec),
            "assumptions": {
                "servo_efficiency": SERVO_EFFICIENCY,
                "idle_w": IDLE_W,
                "usable_capacity": 0.8,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, default=None, help="Microduck walking .npz")
    parser.add_argument("--settle", type=float, default=1.0, help="seconds skipped at start")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    p = biped.BipedParams()
    report: dict[str, Any] = {
        "params": p.to_dict(),
        "mass_budget_kg": sizing.mass_budget(p),
        "ours": our_static(p),
    }
    duck_model = microduck_model()
    report["microduck"] = microduck_static(duck_model)
    if args.reference:
        arrays, _ = recorder.load(args.reference)
        names = [duck_model.joint(j).name for j in duck_model.actuator_trnid[:, 0]]
        walk = sizing.walking_loads(arrays, names, args.settle)
        report["microduck"]["walking"] = walk
        report["requirements"] = requirements(report["ours"], report["microduck"], walk, p)
    text = json.dumps(report, indent=2, default=float)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")

    # Short summary for logs.
    def rounded(d: dict[str, Any], n: int = 3) -> dict[str, Any]:
        return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}

    o = report["ours"]
    print(f"mass {o['mass_kg']:.3f} kg, CoM height {o['com_height_m']:.3f} m")
    for c in o["cases"]:
        print(c["case"], rounded(sizing.by_type(c["torque_nm"])))
    single = report["microduck"]["single"]
    print("microduck", single["case"], rounded(sizing.by_type(single["torque_nm"])))
    if "requirements" in report:
        for t, r in report["requirements"]["joints"].items():
            print(t, rounded(r))
        print("power", rounded(report["requirements"]["power"], 2))


if __name__ == "__main__":
    main()
