"""BAM XL330 M6 port: equations checked against hand-computed values."""

import math

import mujoco
import numpy as np
import pytest

from bipedlab.sim import bam_xl330 as bam

KT, R = bam.M6["kt"], bam.M6["R"]


def test_error_gain_from_firmware_constants():
    assert math.isclose(bam.ERROR_GAIN, (4096 / (2 * math.pi)) / (256 * 885))


def test_force_limit_matches_mjcf_forcerange():
    # 7.4 V * kt / R; the upstream MJCF uses forcerange 0.96 for the same servo.
    assert bam.force_limit(7.4) == pytest.approx(7.4 * KT / R)
    assert bam.force_limit(7.4) == pytest.approx(0.963, abs=1e-3)


def test_small_signal_gain_matches_mjcf_kp():
    # Linearised BAM stiffness = error_gain * kp_fw * vin * kt / R ~ MJCF kp 0.55.
    kp_equiv = bam.ERROR_GAIN * 200 * 7.4 * KT / R
    assert kp_equiv == pytest.approx(0.5545, abs=1e-3)
    volts = bam.compute_control(np.array([0.01]), np.array([0.0]), kp=200, vin=7.4)
    torque = bam.compute_torque(volts, np.array([0.0]))
    assert torque[0] == pytest.approx(kp_equiv * 0.01, rel=1e-9)


def test_pwm_saturates_at_supply_voltage():
    volts = bam.compute_control(np.array([10.0, -10.0]), np.zeros(2), kp=200, vin=7.4)
    np.testing.assert_allclose(volts, [7.4, -7.4])


def test_back_emf_reduces_torque():
    stall = bam.compute_torque(np.array([7.4]), np.array([0.0]))[0]
    moving = bam.compute_torque(np.array([7.4]), np.array([1.0]))[0]
    assert stall - moving == pytest.approx(KT**2 / R)


def test_friction_budget_stopped_motor_side_load():
    p = bam.M6
    fl, damping = bam.compute_frictions(np.array([0.5]), np.array([0.0]), np.array([0.0]))
    expected = (
        p["friction_base"]
        + 0.5 * p["load_friction_motor"]
        + p["friction_stribeck"]  # stribeck coefficient is 1 at rest
        + 0.5 * p["load_friction_motor_stribeck"]
    )  # quadratic term is 0: external torque is 0
    assert fl[0] == pytest.approx(expected)
    assert damping == p["friction_viscous"]


def test_friction_budget_fades_with_speed():
    fast, _ = bam.compute_frictions(np.array([0.5]), np.array([0.0]), np.array([20.0]))
    p = bam.M6
    # Far above the Stribeck velocity only base + direct load friction remain.
    assert fast[0] == pytest.approx(p["friction_base"] + 0.5 * p["load_friction_motor"], rel=1e-6)


def test_reset_keeps_the_start_state():
    # Regression: reset() used to call mj_setConst on the rollout's data, which
    # overwrote the start pose with qpos0 (all joints 0).
    from bipedlab import upstream
    from bipedlab.sim import microduck_policy as mp

    try:
        scene = upstream.microduck_scene()
    except upstream.UpstreamError as exc:
        pytest.skip(str(exc))
    model, data = mp.load_model_for_policy(scene, actuator="bam")
    ctrl = bam.Controller(model)
    mp.upstream_init(model, data)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy()
    ctrl.reset(data)
    np.testing.assert_array_equal(data.qpos, before)
    np.testing.assert_allclose(ctrl.q_target, mp.DEFAULT_POSE)
