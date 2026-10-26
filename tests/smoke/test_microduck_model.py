"""CPU smoke tests on the official Microduck MJCF at the pinned microduck_rl sha.

These record facts we rely on (DoF, gains, keyframes) so an upstream bump that
changes them fails loudly instead of silently breaking sim-to-sim parity.
"""

import numpy as np
import pytest

from bipedlab import upstream
from bipedlab.sim import model as sim_model
from bipedlab.sim import rollout as sim_rollout

pytestmark = pytest.mark.upstream

LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
EXPECTED_ACTUATORS = (
    [f"left_{j}" for j in LEG]
    + ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    + [f"right_{j}" for j in LEG]
)


@pytest.fixture(scope="module")
def microduck(microduck_rl_path):
    path = upstream.microduck_scene("scene.xml")
    return path, *sim_model.load(path)


def test_structure(microduck):
    path, model, _ = microduck
    info = sim_model.describe(model, path)
    assert [a.name for a in info.actuators] == EXPECTED_ACTUATORS
    assert info.timestep == pytest.approx(0.002)
    assert 0.70 < info.total_mass < 0.80
    assert {"STAND", "INIT", "SIT", "FOLD"} <= set(info.keyframes)
    assert {"imu", "left_foot", "right_foot"} <= set(info.sites)


def test_servo_model_matches_research(microduck):
    path, model, _ = microduck
    info = sim_model.describe(model, path)
    for a in info.actuators:
        assert a.kp == pytest.approx(0.55)
        assert a.forcerange == pytest.approx((-0.96, 0.96))


def test_open_loop_stand_is_finite_and_deterministic(microduck):
    _, model, data = microduck
    a = sim_rollout.rollout(model, data, "STAND", 2.0)
    b = sim_rollout.rollout(model, data, "STAND", 2.0)
    assert a.finite
    assert a.state_digest == b.state_digest


def test_open_loop_stand_topples(microduck):
    """Characterization, not a goal: with its soft servo gains and no policy,
    Microduck pitches over in about a second. Balance needs closed-loop control.
    If this starts passing, the model or gains changed upstream."""
    _, model, data = microduck
    result = sim_rollout.rollout(model, data, "STAND", 3.0)
    assert result.fell
    assert result.fall_time_s is not None and result.fall_time_s < 2.0


def test_bam_model_edits(microduck_rl_path):
    import mujoco

    from bipedlab.sim import bam_xl330 as bam

    model = bam.load_model(upstream.microduck_scene(), 0.005)
    assert model.opt.timestep == 0.005
    assert (model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE).all()  # torque motors
    np.testing.assert_allclose(model.actuator_forcerange[:, 1], bam.force_limit())
    ctrl = bam.Controller(model)
    assert len(ctrl.act_ids) == 14
    np.testing.assert_allclose(model.dof_damping[ctrl.dof_idx], 0.0)
    np.testing.assert_allclose(model.dof_armature[ctrl.dof_idx], bam.M6["armature"])
