"""WP4-01 sizing helpers on our own model (no upstream needed)."""

import mujoco
import numpy as np
import pytest

from bipedlab.design import biped, sizing


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(biped.to_mjcf())


def stand_q(model):
    return model.key_qpos[0].copy()


def test_mass_budget_matches_model(model):
    assert sizing.mass_budget(biped.BipedParams())["total"] == pytest.approx(
        model.body_subtreemass[1], abs=1e-9
    )


def test_static_double_support_carries_the_weight(model):
    weight = model.body_subtreemass[1] * sizing.G
    r = sizing.static_loads(
        model, sizing.StaticCase("d", stand_q(model), ["left_foot", "right_foot"])
    )
    assert r["base_residual"] < 1e-9
    fz = [f["fz_n"] for f in r["feet"].values()]
    assert sum(fz) == pytest.approx(weight, rel=1e-6)
    assert fz[0] == pytest.approx(fz[1], rel=1e-6)
    tau = r["torque_nm"]
    assert tau["left_knee"] == pytest.approx(tau["right_knee"], abs=1e-9)


def test_static_single_support_load_and_cop(model):
    weight = model.body_subtreemass[1] * sizing.G
    r = sizing.static_loads(model, sizing.StaticCase("s", stand_q(model), ["left_foot"]))
    foot = r["feet"]["left_foot"]
    assert foot["fz_n"] == pytest.approx(weight, rel=1e-6)
    # CoM sits on the sagittal plane, the foot under the hip: CoP is half the hip
    # spacing inside the stance foot, and the hip roll holds weight x that lever.
    p = biped.BipedParams()
    assert foot["cop_xy_m"][1] == pytest.approx(-p.hip_spacing / 2, abs=1e-6)
    # The stance leg below the hip-roll axis hangs straight under it and adds no lever.
    lever_torque = weight * p.hip_spacing / 2
    assert abs(r["torque_nm"]["left_hip_roll"]) == pytest.approx(lever_torque, rel=1e-6)
    # A deeper crouch loads the knee more.
    deeper = []
    for knee in (0.6, 1.2):
        q = stand_q(model)
        for side in biped.SIDES:
            for j, v in biped.crouch_pose(knee).items():
                q[model.jnt_qposadr[model.joint(f"{side}_{j}").id]] = v
        res = sizing.static_loads(model, sizing.StaticCase("k", q, ["left_foot"]))
        deeper.append(abs(res["torque_nm"]["left_knee"]))
    assert deeper[1] > deeper[0]


def test_walking_loads_and_by_type():
    t = np.arange(0, 3, 0.02)
    tau = np.stack([np.sin(t), 2 * np.cos(t)], axis=1)
    vel = np.stack([np.ones_like(t), -3 * np.ones_like(t)], axis=1)
    out = sizing.walking_loads({"time": t, "joint_torque": tau, "joint_vel": vel},
                               ["left_knee", "right_knee"], settle_s=1.0)  # fmt: skip
    keep = t >= 1.0
    assert out["joints"]["right_knee"]["peak_torque_nm"] == pytest.approx(
        np.abs(2 * np.cos(t[keep])).max()
    )
    assert out["joints"]["left_knee"]["peak_speed_rad_s"] == 1.0
    assert sizing.by_type({"left_knee": -0.3, "right_knee": 0.2, "neck_pitch": 9.0}) == {
        "knee": 0.3
    }
    assert sizing.joint_type("right_ankle_pitch") == "ankle"


def test_battery_runtime():
    # 500 mAh x 7.4 V x 0.8 = 2.96 Wh -> 59.2 min at 3 W.
    assert sizing.battery_runtime_min(500, 7.4, 3.0) == pytest.approx(59.2)
