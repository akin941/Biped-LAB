"""Host side of the runtime command channel (WP4-14).

Velocity commands go to ``biped-runtime --command-udp HOST:PORT`` as 34-byte UDP
packets (format in ``runtime/src/channel.rs``). The runtime's command gate clamps
them, limits their rate and stops the robot 0.5 s after the last packet, so a
sender just keeps sending at 10-50 Hz; stopping to send means stop.

:class:`KeyTeleop` turns key presses into a held command (for ``scripts/teleop.py``);
it has no terminal code, so it is tested on its own.
"""

from __future__ import annotations

import contextlib
import socket
import struct
from dataclasses import dataclass, field

MAGIC = b"BPCM"
VERSION = 1
PACKET = struct.Struct("<4sHI3d")
assert PACKET.size == 34

# Matches GateLimits::default() in runtime/src/gate.rs (the gate clamps anyway).
MAX = (0.3, 0.15, 1.0)


def encode(seq: int, vx: float, vy: float, wz: float) -> bytes:
    return PACKET.pack(MAGIC, VERSION, seq, vx, vy, wz)


def decode(buf: bytes) -> tuple[int, float, float, float]:
    if len(buf) != PACKET.size:
        raise ValueError("not a command packet")
    magic, version, seq, vx, vy, wz = PACKET.unpack(buf)
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a command packet")
    return seq, vx, vy, wz


class CommandSender:
    """Sends commands with increasing sequence numbers from one socket."""

    def __init__(self, host: str, port: int) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        self.seq += 1
        # Nothing listening yet (ICMP unreachable): keep sending.
        with contextlib.suppress(ConnectionRefusedError):
            self.sock.sendto(encode(self.seq, vx, vy, wz), self.addr)

    def close(self) -> None:
        self.sock.close()


@dataclass
class KeyTeleop:
    """w/s forward/back, a/d left/right, q/e turn, space stop; steps of ``step`` x max."""

    step: float = 0.25
    cmd: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    KEYS = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "q": (2, 1), "e": (2, -1)}

    def press(self, key: str) -> tuple[float, float, float]:
        if key == " ":
            self.cmd = [0.0, 0.0, 0.0]
        elif key in self.KEYS:
            axis, sign = self.KEYS[key]
            v = self.cmd[axis] + sign * self.step * MAX[axis]
            self.cmd[axis] = round(max(-MAX[axis], min(MAX[axis], v)), 6)
        return self.command()

    def command(self) -> tuple[float, float, float]:
        return (self.cmd[0], self.cmd[1], self.cmd[2])
