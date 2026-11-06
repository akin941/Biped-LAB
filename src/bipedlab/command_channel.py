"""Host side of the runtime command channel (WP4-14, authenticated in WP4-17).

Velocity commands go to ``biped-runtime --command-udp HOST:PORT [--command-key FILE]``
(format in ``runtime/src/channel.rs``, design in ADR-0014). The sender first asks
for the runtime's session number (hello -> welcome), then sends commands tagged
with HMAC-SHA256 over a pre-shared key. The runtime's command gate clamps them,
limits their rate and stops the robot 0.5 s after the last accepted packet, so a
sender just keeps sending at 10-50 Hz; stopping to send means stop.

Without a key file both sides use an all-zero key, which the runtime allows only
on a loopback address.

:class:`KeyTeleop` turns key presses into a held command (for ``scripts/teleop.py``);
it has no terminal code, so it is tested on its own.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import os
import secrets
import socket
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

VERSION = 2
HELLO, WELCOME, COMMAND = b"BPCH", b"BPCN", b"BPCM"
HELLO_FMT = struct.Struct("<4sHQ")  # 14
WELCOME_FMT = struct.Struct("<4sHQQ")  # 22 + 16 tag
COMMAND_FMT = struct.Struct("<4sHQQ3d")  # 46 + 16 tag
TAG = 16
MIN_KEY = 16
ZERO_KEY = bytes(32)

# Matches GateLimits::default() in runtime/src/gate.rs (the gate clamps anyway).
MAX = (0.3, 0.15, 1.0)


def tag(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()[:TAG]


def encode_hello(client_nonce: int) -> bytes:
    return HELLO_FMT.pack(HELLO, VERSION, client_nonce)


def encode_welcome(key: bytes, client_nonce: int, session: int) -> bytes:
    body = WELCOME_FMT.pack(WELCOME, VERSION, client_nonce, session)
    return body + tag(key, body)


def decode_welcome(key: bytes, buf: bytes, client_nonce: int) -> int:
    """The session number in a welcome, if it is authentic and answers our hello."""
    if len(buf) != WELCOME_FMT.size + TAG:
        raise ValueError("not a welcome")
    body = buf[: WELCOME_FMT.size]
    magic, version, nonce, session = WELCOME_FMT.unpack(body)
    if magic != WELCOME or version != VERSION:
        raise ValueError("not a welcome")
    if not hmac.compare_digest(tag(key, body), buf[WELCOME_FMT.size :]):
        raise ValueError("welcome fails authentication (wrong key?)")
    if nonce != client_nonce:
        raise ValueError("welcome answers another hello")
    return int(session)


def encode_command(key: bytes, session: int, seq: int, vx: float, vy: float, wz: float) -> bytes:
    body = COMMAND_FMT.pack(COMMAND, VERSION, session, seq, vx, vy, wz)
    return body + tag(key, body)


def load_key(path: str | Path) -> bytes:
    key = bytes.fromhex(Path(path).read_text().strip())
    if len(key) < MIN_KEY:
        raise ValueError(f"command key needs at least {MIN_KEY} bytes")
    return key


def new_key_file(path: str | Path, nbytes: int = 32) -> Path:
    """Writes a random hex key readable only by its owner (mode 0600)."""
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secrets.token_hex(nbytes) + "\n")
    return path


class CommandSender:
    """Handshakes with the runtime, then sends tagged commands with rising ``seq``."""

    def __init__(self, host: str, port: int, key: bytes | None = None) -> None:
        self.addr = (host, port)
        self.key = key or ZERO_KEY
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.session: int | None = None
        self.seq = 0

    def connect(self, timeout_s: float = 5.0, retry_s: float = 0.05) -> int:
        """Sends hellos until an authentic welcome arrives; returns the session."""
        nonce = secrets.randbits(64)
        end = time.monotonic() + timeout_s
        self.sock.settimeout(retry_s)
        while time.monotonic() < end:
            with contextlib.suppress(ConnectionRefusedError):
                self.sock.sendto(encode_hello(nonce), self.addr)
            try:
                buf, _ = self.sock.recvfrom(64)
            except (TimeoutError, ConnectionRefusedError):
                continue
            with contextlib.suppress(ValueError):
                self.session = decode_welcome(self.key, buf, nonce)
                return self.session
        raise TimeoutError(f"no authentic welcome from {self.addr} in {timeout_s} s")

    def send(self, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
        if self.session is None:
            raise RuntimeError("connect() first")
        self.seq += 1
        # Nothing listening (ICMP unreachable): keep sending; the gate stops the robot.
        with contextlib.suppress(ConnectionRefusedError):
            self.sock.sendto(
                encode_command(self.key, self.session, self.seq, vx, vy, wz), self.addr
            )

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
