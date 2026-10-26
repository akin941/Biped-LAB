"""Sim side of the runtime bridge (WP4-04, G4.4).

The Rust runtime (``runtime/``, binary ``biped-runtime``) runs the robot-side loop:
command gate -> controller -> safety supervisor. Here MuJoCo plays the robot: it
serves one TCP client on 127.0.0.1, sends a ``Hello`` (joint ranges and the stand
pose), then, in lockstep at the control period, one ``State`` and waits for the
matching ``Action``. The wire format is documented in ``runtime/src/protocol.rs``
and must stay identical to the encoders here.

``torque_enable = 0`` in an action models the servos going limp: the position
gain is set to zero and only the damping term remains.

Fault-injection hooks for the tests: a velocity kick (``push_at``) and a stalled
sim that stops sending states (``stall_at``, exercises the runtime watchdog).
"""

from __future__ import annotations

import json
import math
import socket
import struct
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab.design import biped

MAGIC = b"BPLB"
VERSION = 1
TAG_STATE = 1
TAG_ACTION = 2
# Mirrors runtime/src/safety.rs.
FAULT_TILT = 1 << 0
FAULT_OVERSPEED = 1 << 1
FAULT_NONFINITE_STATE = 1 << 2
FAULT_STATE_TIMEOUT = 1 << 3
EVENT_NONFINITE_TARGET = 1 << 8
EVENT_POSITION_CLAMP = 1 << 9
EVENT_SLEW_LIMIT = 1 << 10

DEFAULT_BINARY = Path(__file__).resolve().parents[2] / "runtime/target/release/biped-runtime"


def encode_hello(control_dt: float, lower: Sequence[float], upper: Sequence[float],
                 default: Sequence[float]) -> bytes:  # fmt: skip
    n = len(default)
    rows = [v for j in range(n) for v in (lower[j], upper[j], default[j])]
    return MAGIC + struct.pack(f"<HHd{3 * n}d", VERSION, n, control_dt, *rows)


def encode_state(step: int, t: float, quat: Sequence[float], gyro: Sequence[float],
                 qpos: Sequence[float], qvel: Sequence[float]) -> bytes:  # fmt: skip
    values = [t, *quat, *gyro, *qpos, *qvel]
    return struct.pack(f"<BQ{len(values)}d", TAG_STATE, step, *values)


def action_size(n: int) -> int:
    return 1 + 8 + 1 + 4 + 8 * n


def decode_action(buf: bytes, n: int) -> tuple[int, bool, int, np.ndarray]:
    tag, step, enable, bits = struct.unpack_from("<BQBI", buf)
    if tag != TAG_ACTION or len(buf) != action_size(n):
        raise ValueError("not an action frame")
    targets = np.array(struct.unpack_from(f"<{n}d", buf, 14))
    return step, bool(enable), bits, targets


def tilt_from_quat(q: Sequence[float]) -> float:
    """Tilt of the body z axis from vertical (rad), quaternion (w, x, y, z)."""
    _, x, y, _ = q
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks, got = [], 0
    while got < size:
        chunk = conn.recv(size - got)
        if not chunk:
            raise ConnectionError("runtime closed the connection")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


@dataclass
class BridgeConfig:
    seconds: float = 10.0
    control_dt: float = 0.02  # 50 Hz
    push_at: float | None = None  # s, add push_dv to the trunk's world velocity
    push_dv: tuple[float, float, float] = (0.0, 1.0, 0.0)
    stall_at: float | None = None  # s, stop sending states (runtime watchdog test)
    action_timeout_s: float = 5.0


class SimBridge:
    def __init__(self, model: mujoco.MjModel | None = None, cfg: BridgeConfig | None = None):
        self.model = model or mujoco.MjModel.from_xml_string(biped.to_mjcf())
        self.cfg = cfg or BridgeConfig()
        m = self.model
        self.joints = m.actuator_trnid[:, 0]
        self.n = len(self.joints)
        self.qadr = m.jnt_qposadr[self.joints]
        self.vadr = m.jnt_dofadr[self.joints]
        self.lower = m.jnt_range[self.joints, 0].copy()
        self.upper = m.jnt_range[self.joints, 1].copy()
        self.default = m.key_ctrl[0].copy()
        self.kp = m.actuator_gainprm[:, 0].copy()
        self.substeps = round(self.cfg.control_dt / m.opt.timestep)
        if not math.isclose(self.substeps * m.opt.timestep, self.cfg.control_dt):
            raise ValueError("control_dt must be a multiple of the model timestep")

    def _torque(self, enable: bool) -> None:
        m = self.model
        m.actuator_gainprm[:, 0] = self.kp if enable else 0.0
        m.actuator_biasprm[:, 1] = -self.kp if enable else 0.0

    def hello(self) -> bytes:
        return encode_hello(self.cfg.control_dt, self.lower, self.upper, self.default)

    def serve(self, conn: socket.socket) -> dict[str, Any]:
        """Run one episode with a connected runtime; returns what the sim saw."""
        m, cfg = self.model, self.cfg
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self._torque(True)
        mujoco.mj_forward(m, d)
        conn.settimeout(cfg.action_timeout_s)
        conn.sendall(self.hello())
        gyro_adr = m.sensor_adr[m.sensor("gyro").id]
        quat_adr = m.sensor_adr[m.sensor("orientation").id]
        out: dict[str, Any] = {
            "steps": 0,
            "max_tilt_deg": 0.0,
            "bits_seen": 0,
            "first_bits_step": {},
            "torque_off_step": None,
            "first_tilt_over_45_step": None,
            "max_target_step_rad": 0.0,
            "max_target_dev_rad": 0.0,
            "stalled": False,
        }
        prev = self.default.copy()
        pushed = False
        wall0 = time.perf_counter()
        step = 0
        while d.time < cfg.seconds - 1e-9:
            step += 1
            if cfg.stall_at is not None and d.time >= cfg.stall_at:
                out["stalled"] = True
                try:  # the runtime should time out and send one torque-off action
                    frame = _recv_exact(conn, action_size(self.n))
                    _, enable, bits, _ = decode_action(frame, self.n)
                    out["after_stall"] = {"torque_enable": enable, "bits": bits}
                except (ConnectionError, TimeoutError, ValueError) as e:
                    out["after_stall"] = {"error": str(e)}
                break
            quat = d.sensordata[quat_adr : quat_adr + 4].copy()
            tilt = tilt_from_quat(quat)
            if tilt > math.radians(45) and out["first_tilt_over_45_step"] is None:
                out["first_tilt_over_45_step"] = step
            out["max_tilt_deg"] = max(out["max_tilt_deg"], math.degrees(tilt))
            conn.sendall(encode_state(step, d.time, quat, d.sensordata[gyro_adr : gyro_adr + 3],
                                      d.qpos[self.qadr], d.qvel[self.vadr]))  # fmt: skip
            got, enable, bits, targets = decode_action(
                _recv_exact(conn, action_size(self.n)), self.n
            )
            if got != step:
                raise ValueError(f"action for step {got}, expected {step}")
            out["bits_seen"] |= bits
            for b in range(32):
                if bits & (1 << b) and str(b) not in out["first_bits_step"]:
                    out["first_bits_step"][str(b)] = step
            if not enable and out["torque_off_step"] is None:
                out["torque_off_step"] = step
            self._torque(enable)
            out["max_target_step_rad"] = max(
                out["max_target_step_rad"], float(np.abs(targets - prev).max())
            )
            out["max_target_dev_rad"] = max(
                out["max_target_dev_rad"], float(np.abs(targets - self.default).max())
            )
            prev = targets
            d.ctrl[:] = targets
            if cfg.push_at is not None and not pushed and d.time >= cfg.push_at:
                d.qvel[0:3] += cfg.push_dv
                pushed = True
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
        out["steps"] = step if not out["stalled"] else step - 1
        out["sim_time_s"] = float(d.time)
        out["wall_s"] = time.perf_counter() - wall0
        out["trunk_height_m"] = float(d.qpos[2])
        self._torque(True)
        return out


def run_with_runtime(
    binary: str | Path = DEFAULT_BINARY,
    runtime_args: Sequence[str] = (),
    cfg: BridgeConfig | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Start ``biped-runtime`` against a fresh sim bridge and return both summaries."""
    bridge = SimBridge(cfg=cfg)
    with socket.create_server(("127.0.0.1", 0)) as server:
        server.settimeout(10.0)
        port = server.getsockname()[1]
        cmd = [str(binary), "--connect", f"127.0.0.1:{port}", *runtime_args]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            conn, _ = server.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sim = bridge.serve(conn)
            stdout, stderr = proc.communicate(timeout=timeout_s)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
    lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
    return {
        "sim": sim,
        "runtime": json.loads(lines[-1]) if lines else None,
        "exit_code": proc.returncode,
        "stderr": stderr.strip(),
    }
