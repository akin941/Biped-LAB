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
