"""Our own biped: parametric MJCF (WP4-02). No upstream needed."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from bipedlab import REPO_ROOT
from bipedlab.design import biped
from bipedlab.sim import rollout as sim_rollout

COMMITTED = REPO_ROOT / "simulation/custom-biped/biped_v1.xml"


def build(p=None):
    model = mujoco.MjModel.from_xml_string(biped.to_mjcf(p))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def budget(p):
    sv = p.servo
    trunk = p.trunk_structure_mass + p.electronics_mass + p.battery_mass
    per_leg = 2 * p.bracket_mass + p.thigh_mass + p.shin_mass + p.foot_mass
    return trunk + 2 * per_leg + 10 * sv.mass


def test_committed_model_is_generated():
    assert COMMITTED.read_text() == biped.to_mjcf()


def test_original_primitive_geometry_only():
    text = biped.to_mjcf()
    assert "mesh" not in text and "file=" not in text  # ADR-0004
    model, _ = build()
    assert set(model.geom_type) <= {mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_BOX}


def test_structure_mass_and_size():
    p = biped.BipedParams()
    model, data = build(p)
    assert model.nu == 10 and model.njnt == 11  # free root + 10 hinges (G4.2: 6-10 DoF)
    names = {model.joint(i).name for i in range(1, model.njnt)}
    assert names == {f"{s}_{j}" for s in biped.SIDES for j in biped.LEG_JOINTS}
    assert model.body_subtreemass[1] == pytest.approx(budget(p), abs=1e-9)
    # G4.2: 20-30 cm tall in the stand pose.
    top = data.qpos[2] + p.trunk_size[2] / 2
    assert 0.20 <= top <= 0.30
    # Left/right mirror: centre of mass on the sagittal plane.
    assert abs(data.subtree_com[1][1]) < 1e-9


def test_stand_keyframe_puts_flat_feet_on_the_floor():
    model, data = build()
    for side in biped.SIDES:
        sole = data.site(f"{side}_foot")
        assert sole.xpos[2] == pytest.approx(0.002, abs=1e-6)
        np.testing.assert_allclose(sole.xmat.reshape(3, 3), np.eye(3), atol=1e-9)
    assert np.all(np.abs(data.qpos[7:]) <= 2.2)


@pytest.mark.parametrize("thigh", [0.055, 0.065, 0.080])
def test_stands_open_loop_across_leg_lengths(thigh):
    p = replace(biped.BipedParams(), thigh=thigh, shin=thigh)
    model, data = build(p)
    result = sim_rollout.rollout(model, data, keyframe="STAND", duration_s=3.0)
    assert result.finite and not result.fell
    assert result.tilt_deg.max() < 5.0
    assert result.trunk_height.min() > biped.stand_height(p, p.stand_knee) - 0.01


def test_ds3218_variant_is_committed_and_about_one_kilogram():
    """G4.5: the DS3218MG build keeps biped_v1's joints and stand pose, ~1 kg."""
    p = biped.BIPED_DS3218
    path = REPO_ROOT / "simulation/custom-biped/biped_ds3218.xml"
    assert path.read_text() == biped.to_mjcf(p)
    model, data = build(p)
    v1, _ = build()
    assert model.nu == v1.nu and model.njnt == v1.njnt
    np.testing.assert_allclose(model.key_qpos[0][7:], v1.key_qpos[0][7:])
    assert model.body_subtreemass[1] == pytest.approx(budget(p), abs=1e-9)
    assert 0.95 <= budget(p) <= 1.15
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], p.servo.torque_limit)
    top = data.qpos[2] + p.trunk_size[2] / 2
    assert 0.20 <= top <= 0.30


def _clearance():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_clearance", REPO_ROOT / "scripts/check_clearance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ds3218b_is_buildable_and_ds3218_is_not():
    """WP5-13: the U-bracket layout clears every same-leg servo pair; the lumped one cannot."""
    check = _clearance().check
    lumped = check("biped_ds3218")
    assert not lumped["ok"]
    assert any({r["a"], r["b"]} == {"left_hip_roll_servo", "left_hip_pitch_servo"}
               for r in lumped["failures"])  # fmt: skip
    built = check("biped_ds3218b")
    assert built["ok"], built["failures"]
    assert all(r["gap_mm"] >= 3.0 for r in built["closest_same_leg"])


def test_box_gap_matches_simple_cases():
    gap = _clearance().box_gap
    eye, h = np.eye(3), np.array([0.01, 0.01, 0.01])
    assert gap(np.zeros(3), eye, h, np.array([0.05, 0, 0]), eye, h) == pytest.approx(0.03)
    assert gap(np.zeros(3), eye, h, np.array([0.015, 0, 0]), eye, h) == pytest.approx(-0.005)


def test_ds3218b_variant_is_committed_and_stands():
    """WP5-13: same joints, ranges, stand pose and mass as biped_ds3218; legs collide."""
    p = biped.BIPED_DS3218B
    path = REPO_ROOT / "simulation/custom-biped/biped_ds3218b.xml"
    assert path.read_text() == biped.to_mjcf(p)
    model, data = build(p)
    ref, _ = build(biped.BIPED_DS3218)
    assert model.nu == ref.nu and model.njnt == ref.njnt
    np.testing.assert_allclose(model.jnt_range, ref.jnt_range)
    np.testing.assert_allclose(model.key_qpos[0][7:], ref.key_qpos[0][7:])
    assert model.body_subtreemass[1] == pytest.approx(budget(p), abs=1e-9)
    top = data.qpos[2] + p.trunk_size[2] / 2
    assert 0.20 <= top <= 0.30  # G4.2
    for side in biped.SIDES:
        assert data.site(f"{side}_foot").xpos[2] == pytest.approx(0.002, abs=1e-6)
    servo = model.geom("left_knee_servo").id
    foot = model.geom("right_foot").id
    assert model.geom_contype[servo] & model.geom_conaffinity[foot]
    floor = model.geom("floor").id
    assert not (model.geom_contype[servo] & model.geom_conaffinity[floor])
    assert not (model.geom_contype[floor] & model.geom_conaffinity[servo])
    result = sim_rollout.rollout(model, data, keyframe="STAND", duration_s=3.0)
    assert result.finite and not result.fell


def test_ds3218b_heavy_only_adds_frame_mass():
    """WP5-15: off-the-shelf bracket masses on the biped_ds3218b geometry, ~1.17 kg."""
    p = biped.BIPED_DS3218B_HEAVY
    path = REPO_ROOT / "simulation/custom-biped/biped_ds3218b_heavy.xml"
    assert path.read_text() == biped.to_mjcf(p)
    model, data = build(p)
    ref, ref_data = build(biped.BIPED_DS3218B)
    np.testing.assert_allclose(model.jnt_range, ref.jnt_range)
    np.testing.assert_allclose(data.qpos, ref_data.qpos)
    assert model.body_subtreemass[1] == pytest.approx(budget(p), abs=1e-9)
    assert 1.15 <= budget(p) <= 1.20
    result = sim_rollout.rollout(model, data, keyframe="STAND", duration_s=3.0)
    assert result.finite and not result.fell
