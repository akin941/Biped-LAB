"""WP4-13: runtime telemetry log reader and conversion to the WP2-01 schema."""

import struct

import numpy as np
import pytest

from bipedlab import runtime_bridge as rb
from bipedlab.telemetry import recorder as rec
from bipedlab.telemetry import report
from bipedlab.telemetry import runtime_log as rl

N = 2


def fake_log(path, periods=50, extra=b""):
    head = rl.MAGIC + struct.pack("<HHd", rl.VERSION, N, 0.02)
    head += struct.pack(f"<{2 * N}d", 2.0, 0.5, 4.0, 1.0)  # (kp, limit) per joint
    recs = np.zeros(periods, rl.record_dtype(N))
    recs["step"] = np.arange(1, periods + 1)
    recs["time"] = recs["step"] * 0.02
    recs["wall"] = np.arange(periods) * 0.02
    recs["quat"][:, 0] = 1.0
    recs["qpos"] = 0.1
    recs["targets"] = 0.3  # error 0.2 rad: kp 2 -> 0.4 N*m, kp 4 -> 0.8 but limit 1.0
    recs["wanted"] = 0.3
    recs["torque_enable"] = 1
    recs["torque_enable"][-5:] = 0
    recs["bits"][10:20] = rb.EVENT_SLEW_LIMIT
    recs["bits"][-5:] = rb.FAULT_TILT
    path.write_bytes(head + recs.tobytes() + extra)
    return path


def test_header_and_record_sizes_match_the_rust_writer():
    # runtime/src/telemetry.rs: header 4+2+2+8+16n, record 8+8+8+4+72+1+8(4n+3)+1+4.
    assert rl.header_size(3) == 16 + 48
    assert rl.record_dtype(3).itemsize == 8 + 8 + 8 + 4 + 72 + 1 + 8 * 15 + 1 + 4


def test_read_drops_a_cut_off_last_record(tmp_path):
    arrays, header = rl.read(fake_log(tmp_path / "a.bplog", extra=b"\x00" * 17))
    assert len(arrays["step"]) == 50 and header["dropped_bytes"] == 17
    assert header["kp"].tolist() == [2.0, 4.0] and header["control_dt"] == 0.02


def test_bad_files_are_rejected(tmp_path):
    bad = tmp_path / "bad.bplog"
    bad.write_bytes(b"NOPE" + b"\x00" * 20)
    with pytest.raises(ValueError, match="magic"):
        rl.read(bad)
    bad.write_bytes(rl.MAGIC + struct.pack("<HH", 9, 1))
    with pytest.raises(ValueError, match="version"):
        rl.read(bad)


def test_convert_estimates_torque_and_counts_supervisor_bits(tmp_path):
    out = rl.convert(fake_log(tmp_path / "a.bplog"), tmp_path / "a.npz", ["l", "r"])
    arrays, sidecar = rec.load(out)
    np.testing.assert_allclose(arrays["joint_torque"][0], [0.4, 0.8])
    assert np.all(arrays["joint_torque"][-5:] == 0)  # torque off
    assert sidecar["meta"]["supervisor_bits_periods"]["slew_limit"] == 10
    assert sidecar["meta"]["supervisor_bits_periods"]["tilt"] == 5
    assert "lin_vel_body" not in arrays and "joint_torque" in sidecar["estimated"]
    s = rec.summarize(arrays)
    assert s["steps"] == 50 and "mean_vx_body_mps" not in s
    assert s["sim_period_max_error_ms"] < 1e-9
    rec.profile(arrays, sidecar["joint_names"])
    html = report.write(out).read_text()
    assert "Güvenlik denetçisi" in html and "Ayak teması" not in html and "Tahmini" in html
