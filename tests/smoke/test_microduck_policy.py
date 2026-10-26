"""Observation/action contract checks for running Microduck ONNX policies."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from bipedlab import upstream  # noqa: E402
from bipedlab.sim import microduck_policy as mp  # noqa: E402
from tests.policy_fixtures import linear_policy  # noqa: E402

pytestmark = [pytest.mark.upstream, pytest.mark.policy]


@pytest.fixture(scope="module")
def scene(microduck_rl_path):
    return upstream.microduck_scene("scene.xml")


def test_obs_layout_at_stand(scene):
    model, data = mp.load_model_for_policy(scene)
    import mujoco

    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    obs = mp.ObservationBuilder(model)(data, np.zeros(14), np.zeros(13))
    s = mp.OBS_SLICES
    assert obs.shape == (61,) and obs.dtype == np.float32
    np.testing.assert_allclose(obs[s["projected_gravity"]], [0, 0, -1], atol=1e-6)
    # STAND keyframe equals DEFAULT_POSE up to rounding of the published constants.
    np.testing.assert_allclose(obs[s["joint_pos_rel"]], 0, atol=1e-4)
    np.testing.assert_allclose(obs[s["base_ang_vel"]], 0, atol=1e-9)


def test_projected_gravity_tracks_tilt(scene):
    import mujoco

    model, data = mp.load_model_for_policy(scene)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    half = np.deg2rad(30) / 2  # pitch 30 degrees about y
    data.qpos[3:7] = [np.cos(half), 0, np.sin(half), 0]
    mujoco.mj_forward(model, data)
    g = mp.ObservationBuilder(model)(data, np.zeros(14), np.zeros(13))[3:6]
    np.testing.assert_allclose(g, [np.sin(np.deg2rad(30)), 0, -np.cos(np.deg2rad(30))], atol=1e-6)


def test_policy_drives_ctrl_and_logs(scene, tmp_path):
    bias = np.full(14, 0.01)
    policy = mp.OnnxPolicy(linear_policy(tmp_path / "p.onnx", np.zeros((61, 14)), bias))
    result, ctrl, extra = mp.run(scene, policy, duration_s=0.2)
    assert len(ctrl.log) == round(0.2 * mp.CONTROL_HZ) == 10
    np.testing.assert_allclose(ctrl.log[-1]["action"], bias, atol=1e-7)
    # last_action fed back into the next observation
    np.testing.assert_allclose(ctrl.log[-1]["obs"][mp.OBS_SLICES["last_action"]], bias, atol=1e-7)
    assert result.finite and extra["max_abs_action"] == pytest.approx(0.01, abs=1e-6)


def test_command_reaches_observation(scene, tmp_path):
    policy = mp.OnnxPolicy(linear_policy(tmp_path / "p.onnx", np.zeros((61, 14)), np.zeros(14)))
    _, ctrl, _ = mp.run(scene, policy, duration_s=0.04, twist=(0.2, 0.0, 0.5))
    np.testing.assert_allclose(ctrl.log[0]["obs"][48:51], [0.2, 0.0, 0.5], atol=1e-7)
    np.testing.assert_allclose(ctrl.log[0]["obs"][51:61], 0)


def test_rejects_wrong_policy_shape(tmp_path):
    path = linear_policy(tmp_path / "bad.onnx", np.zeros((51, 14)), np.zeros(14))
    with pytest.raises(mp.PolicyError, match="51->14"):
        mp.OnnxPolicy(path)


def test_rejects_non_finite_action(tmp_path):
    path = linear_policy(tmp_path / "nan.onnx", np.zeros((61, 14)), np.full(14, np.nan))
    policy = mp.OnnxPolicy(path)
    with pytest.raises(mp.PolicyError, match="non-finite"):
        policy(np.zeros(61, dtype=np.float32))


def test_bench_reports_rates(scene, tmp_path):
    from bipedlab import bench

    policy = linear_policy(tmp_path / "p.onnx", np.zeros((61, 14)), np.zeros(14))
    assert bench.physics_steps_per_s(scene, seconds=0.2) > 100
    assert bench.policy_loop_hz(scene, policy, seconds=0.2) > 50  # faster than real time


@pytest.mark.parametrize("init", ["upstream", "settle"])
def test_init_modes(scene, tmp_path, init):
    policy = mp.OnnxPolicy(linear_policy(tmp_path / "p.onnx", np.zeros((61, 14)), np.zeros(14)))
    result, ctrl, _ = mp.run(scene, policy, duration_s=0.04, init=init)
    rel = ctrl.log[0]["obs"][mp.OBS_SLICES["joint_pos_rel"]]
    assert result.finite
    if init == "upstream":
        np.testing.assert_allclose(rel, 0, atol=0.05)  # starts at DEFAULT_POSE, one step in


def test_unknown_init_rejected(scene, tmp_path):
    policy = mp.OnnxPolicy(linear_policy(tmp_path / "p.onnx", np.zeros((61, 14)), np.zeros(14)))
    with pytest.raises(ValueError, match="init"):
        mp.run(scene, policy, duration_s=0.04, init="bogus")


def test_record_microduck_telemetry(scene, tmp_path):
    from bipedlab.telemetry import recorder as rec

    policy = mp.OnnxPolicy(linear_policy(tmp_path / "p.onnx", np.zeros((61, 14)), np.zeros(14)))
    holder = {}

    def wrap(model, control):
        holder["r"] = rec.Recorder(model, control)
        return holder["r"]

    mp.run(scene, policy, duration_s=0.4, recorder=wrap)
    arrays = holder["r"].arrays()
    assert arrays["joint_pos"].shape == (20, 14)
    assert arrays["foot_contact"].any(axis=0).all()  # Microduck feet found via foot sites


def test_eval_policy_cli_with_postprocess(scene, tmp_path):
    import json

    from bipedlab import cli

    path = linear_policy(tmp_path / "zero.onnx", np.zeros((61, 14)), np.zeros(14))
    out = tmp_path / "report.json"
    assert cli.main(["eval-policy", str(path), "--post", "robotd-walking", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["postprocess"]["action_scale"] == 0.9
    assert report["postprocess"]["legs_lowpass"] == 0.7


def test_bam_rollout_holds_default_pose(scene, tmp_path):
    policy = mp.OnnxPolicy(linear_policy(tmp_path / "z.onnx", np.zeros((61, 14)), np.zeros(14)))
    result, ctrl, _ = mp.run(scene, policy, duration_s=1.0, actuator="bam")
    assert result.finite
    assert ctrl.bam is not None
    np.testing.assert_allclose(ctrl.bam.q_target, mp.DEFAULT_POSE)
    assert 6.0 <= ctrl.bam.last_vin <= 7.4  # voltage sag model active and floored
    assert np.abs(ctrl.bam.last_torque).max() <= 0.97


def test_unknown_actuator_rejected(scene):
    with pytest.raises(ValueError):
        mp.load_model_for_policy(scene, actuator="hydraulic")


def test_bam_command_delay_in_physics_steps(scene):
    from bipedlab.sim import bam_xl330 as bam

    model, data = mp.load_model_for_policy(scene, actuator="bam")
    ctrl = bam.Controller(model, delay_steps=2)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    ctrl.reset(data)
    start = ctrl.q_target.copy()
    ctrl.q_target = start + 0.1
    seen = [ctrl._delayed_target().copy() for _ in range(4)]
    np.testing.assert_allclose(seen[0], start + 0.1)  # nothing older yet
    ctrl2 = bam.Controller(model, delay_steps=2)
    ctrl2.reset(data)
    outs = []
    for k in range(4):
        ctrl2.q_target = start + 0.1 * k
        outs.append(ctrl2._delayed_target()[0] - start[0])
    np.testing.assert_allclose(outs, [0.0, 0.0, 0.0, 0.1], atol=1e-9)  # 2-step lag
    assert len(seen) == 4


def test_joint_vel_lag_returns_previous_step(scene):
    model, data = mp.load_model_for_policy(scene)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    build = mp.ObservationBuilder(model, joint_vel_lag=1)
    s = mp.OBS_SLICES["joint_vel"]
    data.qvel[build.qvel_idx] = 1.0
    first = build(data, np.zeros(14), np.zeros(13))[s]
    data.qvel[build.qvel_idx] = 2.0
    second = build(data, np.zeros(14), np.zeros(13))[s]
    np.testing.assert_allclose(first, 1.0)  # no history: current value
    np.testing.assert_allclose(second, 1.0)  # one control step old


def test_training_match_requires_bam_for_delay(scene, tmp_path):
    policy = mp.OnnxPolicy(linear_policy(tmp_path / "z.onnx", np.zeros((61, 14)), np.zeros(14)))
    with pytest.raises(ValueError):
        mp.run(scene, policy, 0.1, match=mp.TrainingMatch(bam_delay_steps=3))
    result, _, _ = mp.run(
        scene,
        policy,
        0.4,
        actuator="bam",
        match=mp.TrainingMatch(bam_delay_steps=4, vin=6.5, joint_vel_lag=1),
    )
    assert result.finite
