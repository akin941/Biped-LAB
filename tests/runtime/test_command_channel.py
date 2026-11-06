"""WP4-14: commands from a host program reach the robot only through the gate."""

import os
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bipedlab import command_channel as cc
from bipedlab import runtime_bridge as rb
from bipedlab.telemetry import runtime_log as rl

pytestmark = pytest.mark.runtime

BINARY = Path(os.environ.get("BIPEDLAB_RUNTIME_BIN", rb.DEFAULT_BINARY))


@pytest.fixture(scope="module", autouse=True)
def _binary():
    if not BINARY.exists():
        if os.environ.get("BIPEDLAB_REQUIRE_RUNTIME") == "1":
            pytest.fail(f"runtime binary missing: {BINARY}")
        pytest.skip(f"runtime binary not built: {BINARY}")


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def send_for(port: int, cmd, seconds: float, rate: float = 25.0) -> threading.Thread:
    def run():
        tx = cc.CommandSender("127.0.0.1", port)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tx.send(*cmd)
            time.sleep(1.0 / rate)
        tx.close()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def run_with_sender(tmp_path, senders, seconds=3.0):
    port = free_udp_port()
    log = tmp_path / "cmd.bplog"
    threads = [send_for(port, c, s) for c, s in senders]
    r = rb.run_with_runtime(
        BINARY,
        ["--realtime", "--command-udp", f"127.0.0.1:{port}", "--log", str(log)],
        rb.BridgeConfig(seconds=seconds),
    )
    for th in threads:
        th.join()
    assert r["exit_code"] == 0, r["stderr"]
    arrays, _ = rl.read(log)
    return r["runtime"], arrays


def test_host_command_is_ramped_held_and_times_out(tmp_path):
    rt, a = run_with_sender(tmp_path, [((0.2, 0.0, 0.5), 1.5)])
    t, vx, wz = a["time"], a["command"][:, 0], a["command"][:, 2]
    assert rt["channel"]["accepted"] > 20 and rt["channel"]["foreign"] == 0
    assert np.abs(np.diff(vx)).max() <= 0.5 * 0.02 + 1e-9  # gate rate limit
    assert vx.max() == pytest.approx(0.2) and wz.max() == pytest.approx(0.5)
    # Sender stopped near 1.5 s wall: timeout 0.5 s, then ramp to zero at 0.5 m/s^2.
    assert vx[-1] == 0.0 and wz[-1] == 0.0
    last_full = t[np.flatnonzero(np.isclose(vx, 0.2))[-1]]
    assert 1.2 < last_full < 2.4


def test_out_of_range_command_is_clamped_by_the_gate(tmp_path):
    _, a = run_with_sender(tmp_path, [((5.0, -5.0, 9.0), 2.5)])
    cmd = a["command"]
    assert cmd[:, 0].max() == pytest.approx(0.3)
    assert cmd[:, 1].min() == pytest.approx(-0.15)
    assert cmd[:, 2].max() == pytest.approx(1.0)


def test_second_sender_is_ignored(tmp_path):
    port = free_udp_port()
    log = tmp_path / "two.bplog"
    first = send_for(port, (0.1, 0.0, 0.0), 2.5)
    time.sleep(0.3)  # the runtime has locked onto the first sender by now
    second = send_for(port, (0.3, 0.0, 0.0), 2.0)
    r = rb.run_with_runtime(
        BINARY,
        ["--realtime", "--command-udp", f"127.0.0.1:{port}", "--log", str(log)],
        rb.BridgeConfig(seconds=2.5),
    )
    first.join()
    second.join()
    assert r["exit_code"] == 0, r["stderr"]
    arrays, _ = rl.read(log)
    assert r["runtime"]["channel"]["foreign"] > 10
    assert arrays["command"][:, 0].max() == pytest.approx(0.1)
