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
