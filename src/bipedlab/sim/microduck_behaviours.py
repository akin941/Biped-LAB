"""Scripted scenarios for the official Microduck behaviour policies (backlog WP1-08).

Reimplements (does not import) how microduck_rl ``scripts/infer_policy.py`` at the
pinned sha drives each behaviour policy through the 13-D command:

    sitstand     cmd[0] = 1 (sit) / 0 (stand); everything else 0      (:483-487)
    ground_pick  cmd[0:3] = cos(2 pi phase), sin(2 pi phase), 0, phase += dt / 4 s,
                 ends at phase 0.7                                    (:717-731)
    kick_*       all-zero command for 3 s; ball 9 cm ahead, 4.2 cm to the side of
                 the kicking foot, in scene_ball.xml                  (:113-117, :766-781)
    roulade      all-zero command for 2 s                             (:476-480, :1184)

Upstream hands control back to the walking/standing policy when a behaviour ends;
here each run stops at that point and reports the state, so a result describes
the behaviour policy alone. Runs with MJCF position actuators by default or
the BAM XL330 port (``actuator="bam"``), as in ``microduck_policy``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab import upstream
from bipedlab.sim import microduck_policy as mp

CommandFn = Callable[[float], np.ndarray]

GROUND_PICK_PERIOD_S = 4.0
GROUND_PICK_END_PHASE = 0.7
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
BALL_RADIUS = 0.035
UPRIGHT_TILT_DEG = 30.0


def zero_command(t: float) -> np.ndarray:
    return np.zeros(mp.CMD_DIM, dtype=np.float32)


def sitstand_command(sit_s: float) -> CommandFn:
    """Sit flag on for ``sit_s`` seconds, then off (stand back up)."""

    def command(t: float) -> np.ndarray:
        cmd = zero_command(t)
        cmd[0] = 1.0 if t < sit_s else 0.0
        return cmd

    return command


def ground_pick_command(period_s: float = GROUND_PICK_PERIOD_S) -> CommandFn:
    def command(t: float) -> np.ndarray:
        cmd = zero_command(t)
        phase = t / period_s
        cmd[0], cmd[1] = math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase)
        return cmd

    return command


@dataclass(frozen=True)
class Behaviour:
    policy: str
    duration_s: float
    command: CommandFn
    scene: str = "scene.xml"
    ball_side: str | None = None  # "left" / "right": place the kick ball
    expect: str = ""


BEHAVIOURS: dict[str, Behaviour] = {
    "sitstand": Behaviour(
        "alpha_sitstand.onnx",
        8.0,
        sitstand_command(4.0),
        expect="sits (trunk lower) during 0-4 s, stands back up by 8 s",
    ),
    "ground_pick": Behaviour(
        "alpha_ground_pick.onnx",
        GROUND_PICK_PERIOD_S * GROUND_PICK_END_PHASE,
        ground_pick_command(),
        expect="crouches toward the ground and comes back up, stays upright",
    ),
    "kick_left": Behaviour(
        "ball_kick_left.onnx",
        3.0,
        zero_command,
        scene="scene_ball.xml",
        ball_side="left",
        expect="kicks the ball forward with the left foot, stays upright",
    ),
    "kick_right": Behaviour(
        "ball_kick_right.onnx",
        3.0,
        zero_command,
        scene="scene_ball.xml",
        ball_side="right",
        expect="kicks the ball forward with the right foot, stays upright",
    ),
    "roulade": Behaviour(
        "roulade.onnx",
        2.0,
        zero_command,
        expect="forward roll (tilt passes ~180 deg) and ends upright",
    ),
}


def _ball_joint(model: mujoco.MjModel) -> tuple[int, int] | None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if jid < 0:
        return None
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def place_ball(side: str) -> mp.ControlFn:
    """Ball in front of the kicking foot, in the trunk's yaw frame (upstream logic)."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side}")

    def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        ball = _ball_joint(model)
        if ball is None:
            raise mp.PolicyError("scene has no ball_free joint")
        qadr, vadr = ball
        x, y = float(data.qpos[0]), float(data.qpos[1])
        qw, qx, qy, qz = data.qpos[3:7]
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        off_y = -BALL_OFFSET_ABS_Y if side == "right" else BALL_OFFSET_ABS_Y
        bx = x + math.cos(yaw) * BALL_OFFSET_X - math.sin(yaw) * off_y
        by = y + math.sin(yaw) * BALL_OFFSET_X + math.cos(yaw) * off_y
        data.qpos[qadr : qadr + 7] = [bx, by, BALL_RADIUS, 1, 0, 0, 0]
        data.qvel[vadr : vadr + 6] = 0.0

    return init


def run_behaviour(
    name: str, policy_path: str | Path, scene_dir: Path | None = None, actuator: str = "mjcf"
) -> dict[str, Any]:
    """Run one behaviour from the upstream start state; return plain-number metrics."""
    b = BEHAVIOURS[name]
    scene = (scene_dir or upstream.microduck_scene().parent) / b.scene
    after = place_ball(b.ball_side) if b.ball_side else None
    # scene_ball.xml has no keyframes (the ball changes nq); the pose comes from init.
    probe = mujoco.MjModel.from_xml_path(str(scene))
    has_stand = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_KEY, "STAND") >= 0
    keyframe = "STAND" if has_stand else None
    ball_start: np.ndarray | None = None
    ball_end: np.ndarray | None = None

    def recorder(model: mujoco.MjModel, control: mp.ControlFn) -> mp.ControlFn:
        ball = _ball_joint(model)

        def wrapped(m: mujoco.MjModel, d: mujoco.MjData) -> None:
            nonlocal ball_start, ball_end
            if ball is not None:
                pos = d.qpos[ball[0] : ball[0] + 2].copy()
                ball_start = pos if ball_start is None else ball_start
                ball_end = pos
            control(m, d)

        return wrapped

    result, ctrl, extra = mp.run(
        scene,
        mp.OnnxPolicy(policy_path),
        b.duration_s,
        keyframe=keyframe,
        init="upstream",
        recorder=recorder,
        command_fn=b.command,
        after_init=after,
        actuator=actuator,
    )
    final_tilt = float(result.tilt_deg[-1])
    report: dict[str, Any] = {
        "policy": Path(policy_path).name,
        "scene": b.scene,
        "duration_s": b.duration_s,
        "expect": b.expect,
        "upright_at_end": final_tilt < UPRIGHT_TILT_DEG,
        "final_tilt_deg": final_tilt,
        "max_tilt_deg": float(result.tilt_deg.max()),
        "trunk_height_start_m": mp.UPSTREAM_INIT_HEIGHT,
        "trunk_height_min_m": float(result.trunk_height.min()),
        "trunk_height_final_m": float(result.trunk_height[-1]),
        "finite": result.finite,
        **extra,
    }
    if ball_start is not None and ball_end is not None:
        # Sampled at the first and last control step (last 20 ms not included).
        report["ball_displacement_m"] = float(np.linalg.norm(ball_end - ball_start))
    return report


def evaluate(
    lookup: Callable[[str], Path], names: list[str] | None = None, actuator: str = "mjcf"
) -> dict[str, Any]:
    """Run the named behaviours (default: all); ``lookup(file)`` gives the weights."""
    out: dict[str, Any] = {
        "actuator": mp.ACTUATOR_LABELS[actuator],
        "init": "upstream",
        "mujoco": mujoco.__version__,
        "behaviours": {},
    }
    for name in names or list(BEHAVIOURS):
        out["behaviours"][name] = run_behaviour(
            name, lookup(BEHAVIOURS[name].policy), actuator=actuator
        )
    return out
