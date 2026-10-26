"""G4.4: the Rust runtime drives the MuJoCo sim through the socket bridge.

Needs the release binary (``cargo build --release`` in ``runtime/``). Without it
the tests skip, unless ``BIPEDLAB_REQUIRE_RUNTIME=1`` (set in CI), where a missing
binary is a failure.
"""

import math
import os
from pathlib import Path

import pytest

from bipedlab import runtime_bridge as rb

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_RUNTIME_BIN", rb.DEFAULT_BINARY))


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"runtime binary missing: {BINARY}")
        pytest.skip(f"runtime binary not built: {BINARY}")


def run(args=(), **cfg):
    r = rb.run_with_runtime(BINARY, list(args), rb.BridgeConfig(**cfg))
    print(r)
    return r


def test_stands_for_ten_seconds_at_50_hz():
    r = run(seconds=10.0)
    sim, rt = r["sim"], r["runtime"]
    assert r["exit_code"] == 0 and rt["exit"] == "sim_closed"
    assert sim["steps"] == rt["periods"] == 500  # 10 s at 50 Hz, lockstep
    assert rt["latched"] == 0 and sim["bits_seen"] == 0
    assert sim["max_tilt_deg"] < 2.0


def test_tilt_fault_turns_torque_off_in_the_same_period():
    r = run(seconds=5.0, push_at=2.0, push_dv=(0.0, 1.5, 0.0))
    sim = r["sim"]
    assert sim["first_tilt_over_45_step"] is not None
    assert sim["torque_off_step"] == sim["first_tilt_over_45_step"]
    assert sim["bits_seen"] & rb.FAULT_TILT
    assert r["runtime"]["first_fault_step"]["tilt"] == sim["torque_off_step"]


def test_small_push_does_not_trip():
    r = run(seconds=5.0, push_at=2.0, push_dv=(0.0, 0.5, 0.0))
    assert r["runtime"]["latched"] == 0 and r["sim"]["torque_off_step"] is None


def test_nonfinite_targets_are_replaced_by_the_last_safe_ones():
    r = run(["--controller", "nan-at:1.0"], seconds=3.0)
    sim, rt = r["sim"], r["runtime"]
    assert rt["events"]["nonfinite_target"] == 100  # every period from t = 1 s
    assert rt["latched"] == 0 and sim["max_tilt_deg"] < 2.0
    assert sim["max_target_dev_rad"] == 0.0


def test_target_jump_is_slew_limited():
    r = run(["--controller", "jump-at:1.0:3:1.0"], seconds=3.0)  # left knee +1 rad
    sim, rt = r["sim"], r["runtime"]
    assert rt["events"]["slew_limit"] == 5  # 1.0 rad at 10 rad/s x 0.02 s per period
    assert sim["max_target_step_rad"] <= 10.0 * 0.02 + 1e-9
    assert math.isclose(sim["max_target_dev_rad"], 1.0)


def test_watchdog_sends_torque_off_when_states_stop():
    r = run(["--state-timeout-ms", "200"], seconds=3.0, stall_at=1.0)
    assert r["exit_code"] == 3 and r["runtime"]["exit"] == "state_timeout"
    assert r["sim"]["after_stall"] == {"torque_enable": False, "bits": rb.FAULT_STATE_TIMEOUT}


def test_realtime_pacing_holds_50_hz():
    r = run(["--realtime"], seconds=5.0)
    rt = r["runtime"]
    assert rt["periods"] == 250
    assert rt["deadline_misses"] <= 2  # <= 1 % of periods on a shared CI runner
    assert 4.9 < r["sim"]["wall_s"] < 5.5
