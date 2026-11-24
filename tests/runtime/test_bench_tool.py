"""WP5-02: the read-only bench tool runs end to end on the simulated servo bus."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from bipedlab import runtime_bridge as rb

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_BENCH_BIN", rb.DEFAULT_BINARY.with_name("biped-bench")))
MAP = Path(__file__).resolve().parents[2] / "hardware/calibration/biped_v1_joints.example.txt"


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"biped-bench binary missing: {BINARY}")
        pytest.skip(f"biped-bench not built: {BINARY}")


def bench(*args):
    return subprocess.run([BINARY, *map(str, args)], capture_output=True, text=True, timeout=30)


def lines(out):
    return [json.loads(line) for line in out.splitlines()]


def test_scan_read_and_check_on_the_sim_bus():
    r = bench("scan", "--sim")
    assert r.returncode == 0, r.stderr
    assert lines(r.stdout) == [{"found": list(range(1, 11))}]

    r = bench("read", "--sim", "--map", MAP)
    assert r.returncode == 0, r.stderr
    rows = lines(r.stdout)
    assert [row["id"] for row in rows] == list(range(1, 11))
    assert rows[3]["joint"] == "left_knee"
    assert all(row["sim_rad"] == 0.0 and row["voltage"] == 7.4 for row in rows)

    r = bench("check", "--sim", "--map", MAP, "--seconds", "0.3")
    assert r.returncode == 0, r.stderr
    (summary,) = lines(r.stdout)
    assert summary["pass"] is True and summary["reads"] >= 5
    assert summary["timeouts"] == 0 and summary["bad_checksum"] == 0


def test_bad_usage_and_bad_map_are_refused(tmp_path):
    assert bench("scan").returncode == 2
    assert bench("read", "--sim").returncode == 2  # no map
    bad = tmp_path / "map.txt"
    bad.write_text(MAP.read_text().replace(" 10 ", " 9 "))
    r = bench("read", "--sim", "--map", bad)
    assert r.returncode == 2 and "twice" in r.stderr
    r = bench("scan", "--device", tmp_path / "missing-tty")
    assert r.returncode == 2 and "missing-tty" in r.stderr


HELLO = MAP.with_name("biped_v1_sts3215.hello")


def test_offset_is_written_into_a_copy_of_the_map(tmp_path):
    m = tmp_path / "map.txt"
    m.write_text(MAP.read_text())
    r = bench("offset", "--sim", "--map", m, "--joint", "left_knee", "--write")
    assert r.returncode == 0, r.stderr
    (row,) = lines(r.stdout)
    assert row["id"] == 4 and row["written"] is True and row["within_limit"] is True
    assert "left_knee              4   +1  +0.0000" in m.read_text()
    assert bench("offset", "--sim", "--map", m).returncode == 2  # no joint


def test_move_drives_one_joint_inside_its_range(tmp_path):
    common = ["--sim", "--map", MAP, "--hello", HELLO, "--joint", "left_knee"]
    r = bench("move", *common, "--to", "0.3", "--torque-limit", "0.2", "--seconds", "0.2")
    assert r.returncode == 0, r.stderr
    (row,) = lines(r.stdout)
    assert row["towards_target"] is True and abs(row["end_rad"] - 0.3) < 0.002
    r = bench("move", *common, "--to", "3", "--torque-limit", "0.2")
    assert r.returncode == 2 and "outside the joint range" in r.stderr
    r = bench("move", *common, "--to", "0.3", "--torque-limit", "0.8")
    assert r.returncode == 2 and "torque limit" in r.stderr
    assert bench("move", *common, "--to", "0.3").returncode == 2


HELLO_DS3218 = Path(__file__).resolve().parents[2] / "hardware/calibration/biped_ds3218.hello"


def test_pwm_hold_drives_one_joint_through_the_simulated_supervisor():
    from bipedlab import pwm_calib

    r = bench("pwm-hold", "--sim", "--hello", HELLO_DS3218, "--joint", "left_knee",
              "--to", "1.5", "--seconds", "1")  # fmt: skip
    assert r.returncode == 0, r.stderr
    (row,) = lines(r.stdout)
    assert row["estop"] is False and row["frames"] >= 20 and row["bad_frames"] == 0
    # The supervisor ramped all the way to the nominal pulse (same as pwm_calib's plan).
    assert row["pulse_us"] == pwm_calib.firmware_joints()[3].pulse_us(1.5)


def test_pwm_hold_refuses_targets_outside_the_range():
    for joint, to in (("left_knee", "2.5"), ("left_hip_roll", "-0.6"), ("nose", "0")):
        r = bench("pwm-hold", "--sim", "--hello", HELLO_DS3218, "--joint", joint,
                  "--to", to, "--seconds", "0.1")  # fmt: skip
        assert r.returncode == 2, (joint, r.stdout)
