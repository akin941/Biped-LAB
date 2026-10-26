"""Sim bridge protocol and lockstep loop, with a Python stand-in for the runtime."""

import socket
import struct
import threading

import numpy as np

from bipedlab import runtime_bridge as rb


def test_frame_sizes_match_the_rust_protocol():
    # Sizes asserted in runtime/src/protocol.rs tests for n = 2.
    assert len(rb.encode_hello(0.02, [-1, -0.1], [1, 2.2], [0, 0.6])) == 4 + 2 + 2 + 8 + 2 * 24
    assert rb.action_size(2) == 1 + 8 + 1 + 4 + 16
    state = rb.encode_state(7, 0.14, [1, 0, 0, 0], [0.1, -0.2, 0.3], [0.5, 0.6], [-1, 1])
    assert len(state) == 1 + 8 + 8 * (1 + 4 + 3 + 4)


def test_decode_action():
    buf = struct.pack("<BQBI2d", rb.TAG_ACTION, 7, 1, 5, 0.1, 0.2)
    step, enable, bits, targets = rb.decode_action(buf, 2)
    assert (step, enable, bits) == (7, True, 5)
    np.testing.assert_allclose(targets, [0.1, 0.2])


def test_tilt_from_quat():
    h = 2**-0.5
    assert rb.tilt_from_quat([1, 0, 0, 0]) == 0.0
    assert abs(rb.tilt_from_quat([h, h, 0, 0]) - np.pi / 2) < 1e-9
    assert abs(rb.tilt_from_quat([h, 0, 0, h])) < 1e-6


def _python_runtime(conn: socket.socket, n: int, torque_off_after: int | None) -> None:
    """Holds the stand pose from the hello; optionally sends torque off."""
    hello = rb._recv_exact(conn, 4 + 2 + 2 + 8 + 24 * n)
    default = np.array(struct.unpack_from(f"<{3 * n}d", hello, 16))[2::3]
    state_size = 1 + 8 + 8 * (1 + 4 + 3 + 2 * n)
    while True:
        try:
            frame = rb._recv_exact(conn, state_size)
        except ConnectionError:
            return
        step = struct.unpack_from("<Q", frame, 1)[0]
        enable = torque_off_after is None or step <= torque_off_after
        conn.sendall(struct.pack(f"<BQBI{n}d", rb.TAG_ACTION, step, int(enable), 0, *default))


def _run(torque_off_after: int | None = None) -> dict:
    bridge = rb.SimBridge(cfg=rb.BridgeConfig(seconds=2.0))
    a, b = socket.socketpair()
    t = threading.Thread(target=_python_runtime, args=(b, bridge.n, torque_off_after))
    t.start()
    with a:
        out = bridge.serve(a)
    t.join(timeout=10)
    b.close()
    return out


def test_bridge_stands_with_stand_pose_client():
    out = _run()
    assert out["steps"] == 100 and out["torque_off_step"] is None
    assert out["max_tilt_deg"] < 2.0 and out["trunk_height_m"] > 0.19


def test_torque_off_makes_the_robot_sag():
    out = _run(torque_off_after=10)
    assert out["torque_off_step"] == 11
    assert out["trunk_height_m"] < 0.19  # limp joints: the robot sags or falls
