"""WP5-05: command segments and the robot-side metrics of the sim-to-real report."""

import numpy as np
import pytest

from bipedlab.telemetry import sim_vs_real as svr

DT = 0.02


def test_segments_follow_a_ramped_command_and_skip_zero():
    t = np.arange(200) * DT
    cmd = np.zeros((200, 3))
    cmd[50:150, 0] = np.minimum(0.2, 0.005 * np.arange(100))  # gate-like ramp to 0.2
    cmd[150:, 2] = 0.3
    segs = svr.command_segments(t, cmd, tol=0.01)
    assert segs[0][0] == pytest.approx(t[51])  # the first non-zero stretch
    assert segs[-1] == pytest.approx((3.0, 4.0, 0.0, 0.0, 0.3))
    assert max(s[2] for s in segs) == pytest.approx(0.2, abs=0.005)
    assert len(segs) < 30
    args = svr.runtime_command_args(segs[-1:])
    assert args == ["--command", "3:4:0:0:0.3"]
    assert svr.command_segments(t[:0], cmd[:0]) == []


def _log(n=100, on_from=10, wobble=0.0, fault_at=None):
    rng = np.random.default_rng(0)
    targets = np.tile(np.linspace(-0.2, 0.2, 10), (n, 1))
    qpos = np.roll(targets, 1, axis=0) + wobble * rng.normal(size=(n, 10))
    bits = np.zeros(n, np.uint32)
    if fault_at is not None:
        bits[fault_at:] = 1  # tilt
    arrays = {
        "time": np.arange(n) * DT,
        "torque_enable": (np.arange(n) >= on_from).astype(np.uint8),
        "qpos": qpos,
        "targets": targets,
        "quat": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "gyro": np.tile([0.0, 0.0, 0.05], (n, 1)),
        "command": np.tile([0.1, 0.0, 0.0], (n, 1)),
        "bits": bits,
        "bus_voltage": np.full(n, 7.4),
        "max_temperature": np.full(n, 35.0),
    }
    header = {"kp": np.full(10, 10.0), "torque_limit": np.full(10, 1.91)}
    return arrays, header


def test_metrics_and_compare():
    sim = svr.metrics(*_log())
    assert sim["tracking_rms"] == pytest.approx([0.0] * 10, abs=1e-12)
    assert sim["torque_on_fraction"] == pytest.approx(0.9)
    assert sim["yaw_rate_mean"] == pytest.approx(0.05)
    assert sim["latched_faults"] == [] and sim["tilt_max_deg"] == pytest.approx(0.0)
    real = svr.metrics(*_log(wobble=0.01, fault_at=90))
    assert all(0.005 < r < 0.02 for r in real["tracking_rms"])
    assert real["latched_faults"] == ["tilt"]
    c = svr.compare(real, sim)
    assert c["faults_only_real"] == ["tilt"]
    assert c["tracking_ratio"] == [None] * 10  # sim tracked perfectly
    assert c["real_minus_sim"]["yaw_rate_mean"] == pytest.approx(0.0)
    assert "body speed (not measured on the robot)" in c["not_compared"]
    with pytest.raises(ValueError):
        svr.metrics(*_log(n=1))
