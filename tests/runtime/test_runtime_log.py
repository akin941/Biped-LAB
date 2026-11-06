"""WP4-13: the runtime's per-period log matches what the sim bridge saw."""

import os
from pathlib import Path

import numpy as np
import pytest

from bipedlab import runtime_bridge as rb
from bipedlab.telemetry import recorder as rec
from bipedlab.telemetry import runtime_log as rl

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_RUNTIME_BIN", rb.DEFAULT_BINARY))


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"runtime binary missing: {BINARY}")
        pytest.skip(f"runtime binary not built: {BINARY}")


def test_log_records_every_period_as_sent(tmp_path):
    log = tmp_path / "run.bplog"
    r = rb.run_with_runtime(
        BINARY,
        ["--log", str(log), "--controller", "jump-at:1.0:3:0.6", "--command", "0:3:0.2:0:0"],
        rb.BridgeConfig(seconds=3.0, record=True, push_at=2.0, push_dv=(0.0, 2.5, 0.0)),
    )
    assert r["exit_code"] == 0, r["stderr"]
    trace = r["sim"]["trace"]
    arrays, header = rl.read(log)
    assert r["runtime"]["log_records"] == len(trace) == len(arrays["step"]) == 150
    assert r["runtime"]["log_write_errors"] == 0 and header["dropped_bytes"] == 0
    np.testing.assert_array_equal(arrays["step"], np.arange(1, 151))
    np.testing.assert_array_equal(arrays["targets"], np.array([f["targets"] for f in trace]))
    np.testing.assert_array_equal(arrays["qpos"], np.array([f["qpos"] for f in trace]))
    np.testing.assert_array_equal(arrays["bits"], [f["bits"] for f in trace])
    np.testing.assert_array_equal(arrays["torque_enable"], [f["torque_enable"] for f in trace])
    # The jump at 1 s is wanted in full and slew-limited in the sent targets.
    k = int(np.argmax(arrays["time"] >= 1.0 - 1e-9))
    assert arrays["wanted"][k, 3] - arrays["targets"][k - 1, 3] > 0.5
    assert arrays["bits"][k] & rb.EVENT_SLEW_LIMIT
    assert arrays["command"][:, 0].max() > 0.15  # gated command ramps up to 0.2
    out = rl.convert(log, tmp_path / "run.npz")
    tel, sidecar = rec.load(out)
    assert sidecar["meta"]["supervisor_bits_periods"]["slew_limit"] >= 1


def test_unwritable_log_is_refused_before_the_loop(tmp_path):
    r = rb.run_with_runtime(BINARY, ["--log", str(tmp_path / "no/such/dir/x.bplog")],
                            rb.BridgeConfig(seconds=1.0))  # fmt: skip
    assert r["exit_code"] == 2 and "log" in r["stderr"]


def test_watchdog_exit_still_flushes_the_log(tmp_path):
    log = tmp_path / "stall.bplog"
    r = rb.run_with_runtime(BINARY, ["--log", str(log), "--state-timeout-ms", "200"],
                            rb.BridgeConfig(seconds=3.0, stall_at=1.0))  # fmt: skip
    assert r["exit_code"] == 3
    arrays, _ = rl.read(log)
    assert len(arrays["step"]) == r["runtime"]["log_records"] == 50
