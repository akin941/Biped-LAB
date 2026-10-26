import numpy as np
import pytest

from bipedlab import REPO_ROOT
from bipedlab.sim import model as sim_model
from bipedlab.sim import rollout as sim_rollout
from bipedlab.telemetry import recorder as rec

TOY = REPO_ROOT / "simulation/custom-biped/toy_biped_v0.xml"


@pytest.fixture
def recorded():
    model, data = sim_model.load(TOY)
    mujoco_ctrl = sim_rollout.hold_ctrl(model.key("STAND").ctrl.copy())
    r = rec.Recorder(model, mujoco_ctrl, command=lambda: np.array([0.0, 0.0, 0.0]))
    sim_rollout.rollout(model, data, "STAND", 2.0, control=r)
    return model, r


def test_every_channel_recorded_each_step(recorded):
    model, r = recorded
    arrays = r.arrays()
    assert set(arrays) == set(rec.CHANNELS)
    assert all(len(a) == 100 for a in arrays.values())  # 2 s at 50 Hz
    assert arrays["joint_pos"].shape == (100, model.nu)
    assert arrays["foot_contact"].shape == (100, 2)


def test_standing_toy_signals_make_sense(recorded):
    _, r = recorded
    s = rec.summarize(r.arrays())
    assert not s["fell"]
    assert s["max_tilt_deg"] < 5
    assert s["foot_contact_ratio"] == [1.0, 1.0]  # both feet planted the whole time
    assert abs(s["mean_vx_body_mps"]) < 0.01
    assert s["peak_abs_torque_nm"] <= 1.5 + 1e-9  # forcerange of the fixture
    assert s["realtime_factor"] > 1


def test_save_and_load_roundtrip(recorded, tmp_path):
    _, r = recorded
    path = r.save(tmp_path / "run", meta={"env_label": "cloud"})
    arrays, sidecar = rec.load(path)
    assert sidecar["schema_version"] == rec.SCHEMA_VERSION
    assert sidecar["channels"]["tilt"]["unit"] == "deg"
    assert sidecar["meta"]["env_label"] == "cloud"
    np.testing.assert_array_equal(arrays["joint_pos"], r.arrays()["joint_pos"])


def test_fall_shows_up_in_summary(tmp_path):
    model, data = sim_model.load(TOY)
    r = rec.Recorder(model, sim_rollout.hold_ctrl(model.jnt_range[1:, 0].copy()))
    sim_rollout.rollout(model, data, "STAND", 3.0, control=r)
    s = rec.summarize(r.arrays())
    assert s["fell"] and s["first_fall_s"] is not None


def test_html_report_is_self_contained(recorded, tmp_path):
    from bipedlab.telemetry import report

    _, r = recorded
    path = r.save(tmp_path / "run", meta={"env_label": "cloud", "policy": "hold"})
    out = report.write(path)
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "<script" not in text and "http" not in text.split("<main>")[1]  # offline, no JS
    assert text.count("<svg") >= 5
    assert "Gövde eğimi" in text and "cloud" in text


def test_timing_on_synthetic_steps():
    # 5 steps at exactly 20 ms sim period; wall times 1, 2, 3, 30 ms after the first.
    arrays = {
        "time": np.arange(5) * 0.02,
        "wall_dt": np.array([0.0, 0.001, 0.002, 0.003, 0.030]),
    }
    t = rec.timing(arrays)
    assert t["control_period_ms"] == pytest.approx(20.0)
    assert t["sim_period_ms_mean"] == pytest.approx(20.0)
    assert t["sim_period_max_error_ms"] == pytest.approx(0.0, abs=1e-9)
    assert t["wall_step_ms_p50"] == pytest.approx(2.5)
    assert t["wall_step_ms_max"] == pytest.approx(30.0)
    assert t["deadline_miss_ratio"] == pytest.approx(0.25)  # only the 30 ms step
    counts, edges = rec.wall_histogram(arrays, bins=4)
    assert counts.sum() == 4 and len(edges) == 5


def test_recorded_rollout_keeps_exact_control_period(recorded):
    _, r = recorded
    t = rec.timing(r.arrays())
    assert t["sim_period_max_error_ms"] < 1e-6  # decimation is exact
    assert 0.0 <= t["deadline_miss_ratio"] <= 1.0
    assert t["wall_step_ms_p99"] >= t["wall_step_ms_p50"] >= 0.0


def test_html_report_has_timing_histogram(recorded, tmp_path):
    from bipedlab.telemetry import report

    _, r = recorded
    out = report.write(r.save(tmp_path / "run", meta={"env_label": "cloud"}))
    text = out.read_text()
    assert "Adım süresi dağılımı" in text and "Süre aşımı oranı" in text
