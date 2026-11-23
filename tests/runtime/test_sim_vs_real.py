"""WP5-05: the sim-to-real script replays a logged run in the sim bridge."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bipedlab import runtime_bridge as rb

pytestmark = pytest.mark.runtime

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "hardware/calibration"


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not rb.DEFAULT_BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"runtime binary missing: {rb.DEFAULT_BINARY}")
        pytest.skip(f"runtime binary not built: {rb.DEFAULT_BINARY}")


def script(*args):
    cmd = [sys.executable, ROOT / "scripts/sim_vs_real.py", *map(str, args)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=ROOT)


def test_a_sim_log_replayed_in_the_sim_matches_itself(tmp_path):
    real = tmp_path / "real.bplog"
    cfg = rb.BridgeConfig(seconds=2.0)
    args = ["--log", str(real), "--command", "0.5:1.5:0.1:0:0.2"]
    rb.run_with_runtime(runtime_args=args, cfg=cfg)
    assert real.exists()
    r = script(real, "--sim-log", tmp_path / "sim.bplog", "--env-label", "cloud-test")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["real_env"] == "cloud-test" and out["command_segments"] >= 1
    assert out["real"]["periods"] == out["sim"]["periods"] == 100
    assert out["faults_only_real"] == []
    assert abs(out["real_minus_sim"]["tilt_max_deg"]) < 1.0
    assert all(abs(r - 1.0) < 0.2 for r in out["tracking_ratio"] if r is not None)


def test_a_hardware_mode_log_is_accepted(tmp_path):
    real = tmp_path / "hw.bplog"
    hw = [rb.DEFAULT_BINARY, "--hw-sim", "--hello", CAL / "biped_v1_sts3215.hello",
          "--map", CAL / "biped_v1_joints.example.txt", "--imu", "upright",
          "--torque-fraction", "0.3", "--seconds", "1", "--log", real]  # fmt: skip
    assert subprocess.run(list(map(str, hw)), capture_output=True, timeout=30).returncode == 0
    r = script(real, "--env-label", "hw-sim", "--", "--torque-fraction", "0.3")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["real"]["latched_faults"] == [] and out["command_segments"] == 0
