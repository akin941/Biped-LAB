"""Behaviour scenarios (WP1-08) run end to end with a stand-in policy."""

import math

import mujoco
import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from bipedlab import upstream  # noqa: E402
from bipedlab.sim import microduck_behaviours as mb  # noqa: E402
from bipedlab.sim import microduck_policy as mp  # noqa: E402
from tests.policy_fixtures import linear_policy  # noqa: E402

pytestmark = [pytest.mark.upstream, pytest.mark.policy]


def test_sitstand_flag_schedule():
    cmd = mb.sitstand_command(4.0)
    assert cmd(0.0)[0] == 1.0 and cmd(3.99)[0] == 1.0
    assert cmd(4.0)[0] == 0.0 and not cmd(1.0)[1:].any()


def test_ground_pick_phase_encoding():
    cmd = mb.ground_pick_command(4.0)
    np.testing.assert_allclose(cmd(0.0)[:3], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(cmd(1.0)[:3], [0, 1, 0], atol=1e-7)  # quarter period
    assert not cmd(1.0)[3:].any()
    assert mb.BEHAVIOURS["ground_pick"].duration_s == pytest.approx(2.8)


def test_ball_placed_in_front_of_kicking_foot(microduck_rl_path):
    scene = upstream.microduck_scene().parent / "scene_ball.xml"
    model, data = mp.load_model_for_policy(scene)
    mujoco.mj_resetData(model, data)
    mp.upstream_init(model, data)
    for side, sign in (("left", 1), ("right", -1)):
        mb.place_ball(side)(model, data)
        qadr = model.jnt_qposadr[model.joint("ball_free").id]
        np.testing.assert_allclose(
            data.qpos[qadr : qadr + 3], [0.09, sign * 0.042, 0.035], atol=1e-9
        )


def test_place_ball_rejects_unknown_side():
    with pytest.raises(ValueError):
        mb.place_ball("middle")


@pytest.fixture(scope="module")
def zero_policy(tmp_path_factory):
    path = tmp_path_factory.mktemp("p") / "zero.onnx"
    return linear_policy(path, np.zeros((61, 14)), np.zeros(14))


@pytest.mark.parametrize("name", list(mb.BEHAVIOURS))
def test_every_behaviour_runs(name, zero_policy, microduck_rl_path):
    report = mb.run_behaviour(name, zero_policy)
    assert report["finite"]
    assert math.isfinite(report["trunk_height_min_m"])
    assert isinstance(report["upright_at_end"], bool)
    assert ("ball_displacement_m" in report) == name.startswith("kick")


def test_eval_behaviours_cli_subset(zero_policy, microduck_rl_path, monkeypatch, tmp_path):
    import json

    from bipedlab import cli, policies

    monkeypatch.setattr(policies, "verified_path", lambda name: zero_policy)
    out = tmp_path / "b.json"
    assert cli.main(["eval-behaviours", "--only", "roulade", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert list(report["behaviours"]) == ["roulade"]
    assert report["actuator"].startswith("mjcf-position")
