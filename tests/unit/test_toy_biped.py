"""Fast checks on our own fixture model (no upstream needed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from bipedlab import REPO_ROOT, cli
from bipedlab.sim import model as sim_model
from bipedlab.sim import rollout as sim_rollout

TOY = REPO_ROOT / "simulation/custom-biped/toy_biped_v0.xml"


@pytest.fixture
def toy():
    return sim_model.load(TOY)


def test_describe_toy(toy):
    model, _ = toy
    info = sim_model.describe(model, TOY)
    assert info.nu == 8
    assert info.timestep == pytest.approx(0.002)
    assert info.total_mass == pytest.approx(0.8, abs=1e-9)
    assert info.joints[0].type == "free"
    assert {a.joint for a in info.actuators} == {j.name for j in info.joints[1:]}
    assert all(a.forcerange == (-1.5, 1.5) for a in info.actuators)
    assert info.keyframes == ["STAND"]


def test_toy_stands_open_loop(toy):
    result = sim_rollout.rollout(*toy, keyframe="STAND", duration_s=5.0)
    assert result.finite
    assert not result.fell
    assert result.tilt_deg.max() < 5.0
    assert result.trunk_height.min() > 0.18


def test_rollout_is_deterministic(toy):
    a = sim_rollout.rollout(*toy, keyframe="STAND", duration_s=1.0)
    b = sim_rollout.rollout(*toy, keyframe="STAND", duration_s=1.0)
    assert a.state_digest == b.state_digest
    np.testing.assert_array_equal(a.trunk_height, b.trunk_height)


def test_fall_is_detected(toy):
    model, data = toy
    # Collapse every joint to its lower limit: the robot must topple, and we must notice.
    limp = model.jnt_range[1:, 0].copy()
    result = sim_rollout.rollout(model, data, "STAND", 3.0, control=sim_rollout.hold_ctrl(limp))
    assert result.fell
    assert result.fall_time_s is not None and result.fall_time_s < 3.0


def test_rejects_bad_control_rate(toy):
    with pytest.raises(ValueError, match="divisor"):
        sim_rollout.rollout(*toy, keyframe="STAND", duration_s=0.1, control_hz=33.0)


@pytest.mark.parametrize(
    ("quat", "expected"),
    [((1, 0, 0, 0), 0.0), ((np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0), 90.0)],
)
def test_tilt(quat, expected):
    assert sim_rollout.tilt_deg(np.array(quat)) == pytest.approx(expected, abs=1e-6)


def test_cli_stand_json(capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    assert cli.main(["stand", "toy", "--seconds", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fell"] is False
    assert Path(out["model"]).name == "toy_biped_v0.xml"
