"""Robot-runtime action stage (PostProcess) against hand-computed values."""

import numpy as np
import pytest

from bipedlab.sim import microduck_policy as mp


def test_default_is_identity():
    post = mp.PostProcess()
    action = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    assert post.is_identity
    np.testing.assert_allclose(post.targets(action, None), mp.DEFAULT_POSE + action)
    np.testing.assert_allclose(post.targets(action, np.zeros(14)), mp.DEFAULT_POSE + action)


def test_scale_and_voltage_compensation():
    assert mp.PostProcess(action_scale=0.9).scale == pytest.approx(0.9)
    assert mp.PostProcess(action_scale=0.9, supply_voltage=7.4).scale == pytest.approx(0.9)
    assert mp.PostProcess(supply_voltage=6.5).scale == pytest.approx(7.4 / 6.5)
    # robotd clamps the measured voltage to 6.0..9.5 V.
    assert mp.PostProcess(supply_voltage=3.0).scale == pytest.approx(7.4 / 6.0)
    assert mp.PostProcess(supply_voltage=12.0).scale == pytest.approx(7.4 / 9.5)


def test_lowpass_blends_head_and_legs_separately():
    post = mp.PostProcess(action_scale=1.0, head_lowpass=0.5, legs_lowpass=0.7)
    action = np.ones(14, dtype=np.float32)
    first = post.targets(action, None)
    np.testing.assert_allclose(first, mp.DEFAULT_POSE + 1.0)  # no history: unfiltered
    previous = mp.DEFAULT_POSE.copy()
    out = post.targets(action, previous)
    np.testing.assert_allclose(out[mp.HEAD_JOINTS], mp.DEFAULT_POSE[mp.HEAD_JOINTS] + 0.5)
    np.testing.assert_allclose(out[mp.LEG_JOINTS], mp.DEFAULT_POSE[mp.LEG_JOINTS] + 0.7, rtol=1e-6)


def test_head_and_leg_index_sets_partition_joints():
    joints = set(range(14)[mp.HEAD_JOINTS]) | set(mp.LEG_JOINTS.tolist())
    assert joints == set(range(14)) and len(mp.LEG_JOINTS) == 10


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError):
        mp.PostProcess(legs_lowpass=alpha)


def test_controller_filters_ctrl_but_feeds_back_raw_action():
    class Data:
        ctrl = np.zeros(14)

    action = np.full(14, 0.2, dtype=np.float32)
    ctrl = mp.PolicyController(
        policy=lambda obs: action,
        observe=lambda data, last, cmd: np.zeros(61, dtype=np.float32),  # type: ignore[arg-type]
        post=mp.ROBOTD_WALKING,
    )
    data = Data()
    ctrl(None, data)  # type: ignore[arg-type]
    np.testing.assert_allclose(data.ctrl, mp.DEFAULT_POSE + 0.9 * 0.2, rtol=1e-6)
    ctrl(None, data)  # type: ignore[arg-type]
    np.testing.assert_allclose(data.ctrl, mp.DEFAULT_POSE + 0.9 * 0.2, rtol=1e-6)  # steady
    np.testing.assert_array_equal(ctrl.last_action, action)
    assert mp.POSTPROCESS_PRESETS["robotd-walking"] is mp.ROBOTD_WALKING
