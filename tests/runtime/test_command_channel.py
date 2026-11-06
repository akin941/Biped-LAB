"""WP4-14/16: authenticated host commands reach the robot only through the gate."""

import contextlib
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


def send_for(port, cmd, seconds, rate=25.0, key=None, sent=None, after=None) -> threading.Thread:
    """Handshake, then send ``cmd`` for ``seconds``; raw packets go to ``sent``.

    ``after``: a list to wait on until it holds 10 packets (another sender's ``sent``).
    """

    def run():
        while after is not None and len(after) < 10:
            time.sleep(0.01)
        tx = cc.CommandSender("127.0.0.1", port, key)
        try:
            tx.connect(timeout_s=5.0)
        except TimeoutError:
            tx.close()
            return
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tx.send(*cmd)
            if sent is not None:
                sent.append(cc.encode_command(tx.key, tx.session, tx.seq, *cmd))
            time.sleep(1.0 / rate)
        tx.close()

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def replay_for(port, packets, seconds, rate=25.0) -> threading.Thread:
    """Sends recorded packets as they are, with no handshake (an attacker's replay)."""

    def run():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            end = time.monotonic() + seconds
            i = 0
            while time.monotonic() < end:
                with contextlib.suppress(ConnectionRefusedError):
                    s.sendto(packets[i % len(packets)], ("127.0.0.1", port))
                i += 1
                time.sleep(1.0 / rate)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def runtime(tmp_path, port, extra=(), seconds=3.0):
    log = tmp_path / f"cmd-{port}.bplog"
    r = rb.run_with_runtime(
        BINARY,
        ["--realtime", "--command-udp", f"127.0.0.1:{port}", "--log", str(log), *extra],
        rb.BridgeConfig(seconds=seconds),
    )
    assert r["exit_code"] == 0, r["stderr"]
    arrays, _ = rl.read(log)
    return r["runtime"], arrays


def run_with_sender(tmp_path, senders, seconds=3.0):
    port = free_udp_port()
    threads = [send_for(port, c, s) for c, s in senders]
    out = runtime(tmp_path, port, seconds=seconds)
    for th in threads:
        th.join()
    return out


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
    sent: list[bytes] = []
    first = send_for(port, (0.1, 0.0, 0.0), 2.5, sent=sent)
    # Starts once the first sender has sent 10 commands, so the runtime is locked on it.
    second = send_for(port, (0.3, 0.0, 0.0), 2.0, after=sent)
    rt, arrays = runtime(tmp_path, port, seconds=2.5)
    first.join()
    second.join()
    assert rt["channel"]["foreign"] > 10
    assert arrays["command"][:, 0].max() == pytest.approx(0.1)


def test_key_file_command_is_accepted(tmp_path):
    keyfile = cc.new_key_file(tmp_path / "k.hex")
    port = free_udp_port()
    th = send_for(port, (0.2, 0.0, 0.0), 2.0, key=cc.load_key(keyfile))
    rt, a = runtime(tmp_path, port, ["--command-key", str(keyfile)], seconds=2.5)
    th.join()
    assert rt["channel"]["accepted"] > 20 and rt["channel"]["unauthenticated"] == 0
    assert a["command"][:, 0].max() == pytest.approx(0.2)


def test_wrong_key_moves_nothing(tmp_path):
    keyfile = cc.new_key_file(tmp_path / "k.hex")
    wrong = cc.load_key(cc.new_key_file(tmp_path / "wrong.hex"))
    port = free_udp_port()
    sent: list[bytes] = []
    th = send_for(port, (0.2, 0.0, 0.0), 2.0, key=wrong, sent=sent)
    rt, a = runtime(tmp_path, port, ["--command-key", str(keyfile)], seconds=2.5)
    th.join()
    # The welcome fails the sender's check, so it never sends a command.
    assert sent == [] and rt["channel"]["hellos"] > 0
    assert rt["channel"]["accepted"] == 0
    assert np.all(a["command"] == 0.0)


def test_replayed_packets_from_an_old_session_are_rejected(tmp_path):
    keyfile = cc.new_key_file(tmp_path / "k.hex")
    key = cc.load_key(keyfile)
    port = free_udp_port()
    sent: list[bytes] = []
    th = send_for(port, (0.2, 0.0, 0.0), 1.5, key=key, sent=sent)
    rt1, _ = runtime(tmp_path, port, ["--command-key", str(keyfile)], seconds=2.0)
    th.join()
    assert rt1["channel"]["accepted"] > 10 and len(sent) > 10
    # Same key, new runtime: the recorded packets carry the old session number.
    port = free_udp_port()
    th = replay_for(port, sent, 2.0)
    rt2, a = runtime(tmp_path, port, ["--command-key", str(keyfile)], seconds=2.5)
    th.join()
    assert rt2["channel"]["accepted"] == 0 and rt2["channel"]["unauthenticated"] > 10
    assert np.all(a["command"] == 0.0)


def test_keyless_channel_refuses_a_network_address():
    r = rb.run_with_runtime(
        BINARY, ["--command-udp", f"0.0.0.0:{free_udp_port()}"], rb.BridgeConfig(seconds=1.0)
    )
    assert r["exit_code"] == 2
    assert "needs --command-key" in r["stderr"]
