"""WP5-18: bench calibration of the DS3218MG pulse mapping."""

import math
import subprocess
import sys
from pathlib import Path

import pytest

from bipedlab import pwm_calib as pc

ROOT = Path(__file__).resolve().parents[2]


def test_firmware_joints_match_the_rust_mapping():
    joints = pc.firmware_joints()
    assert len(joints) == 10
    knee = joints[3]
    assert knee.offset == pytest.approx(-1.05)
    assert knee.pulse_us(1.05) == 1500
    assert joints[0].pulse_us(math.pi / 2) == 2500
    assert joints[8] == knee
    for j in joints:  # the same check as joints.rs's unit test
        assert 500 <= j.pulse_us(j.lower) < 1500 < j.pulse_us(j.upper) <= 2500


def test_plan_stays_inside_the_middle_of_each_range():
    rows = pc.plan()
    assert len(rows) == 50
    for row, j in zip(rows, [j for j in pc.firmware_joints() for _ in range(5)], strict=True):
        margin = 0.19 * (j.upper - j.lower)
        assert j.lower + margin <= row["commanded_rad"] <= j.upper - margin
        assert 500 < row["pulse_us"] < 2500


def test_calibration_recovers_a_mirrored_servo_and_a_horn_offset():
    gain = {"right_hip_roll": -1.0, "left_knee": 1.08, "right_ankle_pitch": 0.95}
    bias = {"left_hip_yaw": math.radians(5.0), "left_knee": math.radians(-4.0)}
    report = pc.calibrate(pc.synthetic_measurements(gain, bias, noise_deg=0.3))
    assert report["ok"], report["problems"]
    joints = pc.firmware_joints()
    for k, name in enumerate(pc.JOINT_NAMES):
        new = pc.JointPwm(**report["joints"][name]["pwm"])
        g, b = gain.get(name, 1.0), bias.get(name, 0.0)
        for q in (joints[k].lower * 0.5, 0.0, joints[k].upper * 0.5):
            # The pulse the corrected mapping sends; the nominal mapping reads it back as
            # q_cmd, and the real servo then sits at g * q_cmd + b, which must be q.
            q_cmd = joints[k].angle(new.pulse_us(q))
            assert g * q_cmd + b == pytest.approx(q, abs=math.radians(1.0)), (name, q)


def test_refuses_a_wrong_tooth_a_wrong_servo_and_missing_joints():
    meas = pc.synthetic_measurements({"left_hip_pitch": 1.4}, {"right_knee": math.radians(12.0)})
    report = pc.calibrate(meas)
    text = " ".join(report["problems"])
    assert not report["ok"]
    assert "left_hip_pitch: gain" in text and "right_knee: bias" in text
    del meas["left_hip_yaw"]
    assert pc.calibrate(meas)["missing"] == ["left_hip_yaw"]


def test_refuses_noisy_readings_and_too_few_points():
    noisy = pc.synthetic_measurements({}, {}, noise_deg=5.0, seed=1)
    assert any("residual" in p for p in pc.calibrate(noisy)["problems"])
    two = {n: s[:2] for n, s in pc.synthetic_measurements({}, {}).items()}
    assert any("need 3" in p for p in pc.calibrate(two)["problems"])
    with pytest.raises(ValueError):
        pc.fit([(0.1, 0.1), (0.1, 0.12)])


def test_cli_plan_and_fit_round_trip(tmp_path):
    script = ROOT / "scripts/pwm_calibration.py"
    plan = subprocess.run(
        [sys.executable, str(script), "plan"], capture_output=True, text=True, check=True
    ).stdout
    rows = [line.split(",") for line in plan.strip().splitlines()]
    assert rows[0] == ["joint", "commanded_rad", "measured_deg", "pulse_us"]
    filled = ["joint,commanded_rad,measured_deg"] + [
        f"{j},{q},{math.degrees(float(q)) + 1.0}" for j, q, _, _ in rows[1:]
    ]
    path = tmp_path / "readings.csv"
    path.write_text("\n".join(filled) + "\n")
    out = subprocess.run(
        [sys.executable, str(script), "fit", str(path), "--out", str(tmp_path / "r.json")],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "biped_ds3218_calibrated" in out.stdout
    assert '"bias_deg": 1.0' in (tmp_path / "r.json").read_text()


def test_a_small_unreachable_range_end_is_reported_not_refused():
    small = pc.calibrate(pc.synthetic_measurements({}, {"left_hip_pitch": math.radians(3.0)}))
    assert small["ok"] and small["joints"]["left_hip_pitch"]["unreachable_deg"] > 3.0
    big = pc.synthetic_measurements({"left_hip_pitch": 0.88}, {"left_hip_pitch": 0.0})
    assert any("past the servo travel" in p for p in pc.calibrate(big)["problems"])
