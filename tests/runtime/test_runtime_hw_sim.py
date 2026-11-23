"""WP5-03: biped-runtime's hardware mode runs end to end on the simulated servo bus."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from bipedlab import runtime_bridge as rb

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_RUNTIME_BIN", rb.DEFAULT_BINARY))
CAL = Path(__file__).resolve().parents[2] / "hardware/calibration"
HW = [
    "--hello", CAL / "biped_v1_sts3215.hello",
    "--map", CAL / "biped_v1_joints.example.txt",
    "--imu", "upright",
]  # fmt: skip


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"runtime binary missing: {BINARY}")
        pytest.skip(f"runtime binary not built: {BINARY}")


def runtime(*args):
    return subprocess.run([BINARY, *map(str, args)], capture_output=True, text=True, timeout=30)


def test_stand_on_the_sim_bus_paces_and_ends_cleanly(tmp_path):
    log = tmp_path / "hw.bplog"
    r = runtime("--hw-sim", *HW, "--torque-fraction", "0.3", "--seconds", "1", "--log", log)
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["exit"] == "duration" and s["latched"] == 0
    assert 40 <= s["periods"] <= 51  # paced at 50 Hz, not as fast as possible
    assert s["log_records"] == s["periods"] and log.stat().st_size > 0


def test_hardware_mode_refuses_unsafe_or_incomplete_setups(tmp_path):
    base = ["--hw-sim", *HW]
    assert "--torque-fraction" in runtime(*base, "--seconds", "1").stderr
    no_imu = [a for a in base if a not in ("--imu", "upright")]
    assert "--imu upright" in runtime(*no_imu, "--torque-fraction", "0.3").stderr
    assert runtime("--connect", "127.0.0.1:1", "--hw-sim").returncode == 2
    bad = tmp_path / "map.txt"
    bad.write_text((CAL / "biped_v1_joints.example.txt").read_text().replace(" 10 ", " 9 "))
    swapped = [str(bad) if str(a).endswith("example.txt") else a for a in base]
    r = runtime(*swapped, "--torque-fraction", "0.3")
    assert r.returncode == 2 and "twice" in r.stderr


def test_mpu6050_imu_on_a_simulated_chip_runs_cleanly():
    """WP5-07: --imu mpu6050:sim runs the real driver and filter on a level simulated chip."""
    hw = [a if a != "upright" else "mpu6050:sim" for a in HW]
    r = runtime("--hw-sim", *hw, "--torque-fraction", "0.3", "--seconds", "1")
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["exit"] == "duration" and s["latched"] == 0 and s["periods"] >= 40


def test_mpu6050_imu_specs_are_checked():
    def with_imu(spec):
        hw = [a if a != "upright" else spec for a in HW]
        return runtime("--hw-sim", *hw, "--torque-fraction", "0.3", "--seconds", "1")

    r = with_imu("mpu6050:")
    assert r.returncode == 2 and "needs a device" in r.stderr
    r = with_imu("mpu6050:sim:x,x,z")
    assert r.returncode == 2 and "once" in r.stderr
    r = with_imu("mpu6050:/dev/no-such-i2c")
    assert r.returncode == 2 and "imu" in r.stderr


PWM = [
    "--servos", "pwm",
    "--hello", CAL / "biped_ds3218.hello",
    "--imu", "upright",
    "--joint-feedback", "commanded",
]  # fmt: skip


def test_pwm_link_to_the_simulated_supervisor_mcu_runs_cleanly(tmp_path):
    """WP5-09: the DS3218MG build's UART link, with the MCU's supervisor-core logic behind it."""
    log = tmp_path / "pwm.bplog"
    r = runtime("--hw-sim", *PWM, "--torque-fraction", "0.3", "--seconds", "1", "--log", log)
    assert r.returncode == 0, r.stderr
    s = json.loads(r.stdout)
    assert s["exit"] == "duration" and s["latched"] == 0
    assert 40 <= s["periods"] <= 51 and s["log_records"] == s["periods"]


def test_pwm_link_setups_are_checked(tmp_path):
    base = ["--hw-sim", *PWM, "--torque-fraction", "0.3", "--seconds", "1"]
    r = runtime(*base, "--map", CAL / "biped_v1_joints.example.txt")
    assert r.returncode == 2 and "own calibration" in r.stderr
    r = runtime(*[a if a != "pwm" else "can" for a in base])
    assert r.returncode == 2 and "sts or pwm" in r.stderr
    r = runtime("--connect", "127.0.0.1:1", "--servos", "pwm")
    assert r.returncode == 2 and "hardware mode" in r.stderr
    r = runtime("--device", tmp_path / "no-tty", *base[1:])
    assert r.returncode == 2 and "no-tty" in r.stderr
