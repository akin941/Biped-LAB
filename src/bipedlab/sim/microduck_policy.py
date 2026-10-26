"""Run official Microduck ONNX policies in CPU MuJoCo.

Reimplements (does not import) the observation/action contract of
pollen-robotics/microduck_rl ``scripts/infer_policy.py`` at the pinned sha:

    obs (61) = base_ang_vel(3)     imu_ang_vel gyro sensor, body frame
             + projected_gravity(3) world -z rotated into trunk_base frame
             + joint_pos_rel(14)    qpos - DEFAULT_POSE
             + joint_vel(14)
             + last_action(14)
             + command(13)          twist[vx, vy, wz](3) + head(4) + body pose(6)
    action (14): ctrl target = DEFAULT_POSE + action * 1.0
    timing: physics dt 0.005 s, decimation 4 -> 50 Hz

Joint order is the MJCF actuator order (left leg 0-4, neck/head 5-8, right leg 9-13).

Actuator caveat: training uses the BAM XL330 model; here we drive MuJoCo's
position actuators (upstream's ``--no-bam`` path). That is a known sim-to-sim
gap, so results from this module are labelled ``actuator="mjcf-position"``.
See docs/research/microduck_rl.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab.sim import rollout as sim_rollout

ControlFn = Callable[[mujoco.MjModel, mujoco.MjData], None]

OBS_DIM = 61
ACT_DIM = 14
CMD_DIM = 13
PHYSICS_DT = 0.005
DECIMATION = 4
CONTROL_HZ = 1.0 / (PHYSICS_DT * DECIMATION)

# microduck_rl scripts/infer_policy.py:126-140 (STAND2 pose).
DEFAULT_POSE = np.array(
    [
        0.0, -0.0873, -0.4579, -0.0049, 0.4530,  # left leg
        0.3491, 0.3491, 0.0, 0.0,  # neck_pitch, head_pitch, head_yaw, head_roll
        0.0, 0.0873, 0.4579, 0.0049, -0.4530,  # right leg
    ],
    dtype=np.float32,
)  # fmt: skip

OBS_SLICES = {
    "base_ang_vel": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "joint_pos_rel": slice(6, 20),
    "joint_vel": slice(20, 34),
    "last_action": slice(34, 48),
    "twist_cmd": slice(48, 51),
    "head_cmd": slice(51, 55),
    "body_cmd": slice(55, 61),
}


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, xyz = quat_wxyz[0], quat_wxyz[1:4]
    t = np.cross(xyz, vec) * 2.0
    return np.asarray(vec - w * t + np.cross(xyz, t))


class PolicyError(RuntimeError):
    pass


@dataclass
class ObservationBuilder:
    model: mujoco.MjModel
    qpos_idx: np.ndarray = field(init=False)
    qvel_idx: np.ndarray = field(init=False)
    gyro_adr: int = field(init=False)
    trunk_id: int = field(init=False)

    def __post_init__(self) -> None:
        m = self.model
        if m.nu != ACT_DIM:
            raise PolicyError(f"expected {ACT_DIM} actuators, model has {m.nu}")
        joints = m.actuator_trnid[:, 0]
        self.qpos_idx = m.jnt_qposadr[joints].astype(int)
        self.qvel_idx = m.jnt_dofadr[joints].astype(int)
        sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if sensor < 0 or body < 0:
            raise PolicyError("model lacks imu_ang_vel sensor or trunk_base body")
        self.gyro_adr = int(m.sensor_adr[sensor])
        self.trunk_id = body

    def __call__(
        self, data: mujoco.MjData, last_action: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        gravity = quat_rotate_inverse(data.xquat[self.trunk_id], np.array([0.0, 0.0, -1.0]))
        obs = np.concatenate(
            [
                data.sensordata[self.gyro_adr : self.gyro_adr + 3],
                gravity,
                data.qpos[self.qpos_idx] - DEFAULT_POSE,
                data.qvel[self.qvel_idx],
                last_action,
                command,
            ]
        ).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs


class OnnxPolicy:
    """CPU-only ONNX Runtime session with shape and finiteness checks."""

    def __init__(self, path: str | Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise PolicyError(f"expected 1 input and >=1 output, got {len(inputs)}/{len(outputs)}")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        in_dim, out_dim = inputs[0].shape[-1], outputs[0].shape[-1]
        if in_dim != OBS_DIM or out_dim != ACT_DIM:
            raise PolicyError(f"policy is {in_dim}->{out_dim}, contract is {OBS_DIM}->{ACT_DIM}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self.session.run([self.output_name], {self.input_name: obs[None, :]})[0]
        action = np.asarray(out, dtype=np.float32).reshape(ACT_DIM)
        if not np.isfinite(action).all():
            raise PolicyError("policy produced a non-finite action")
        return action


@dataclass
class PolicyController:
    """Closed-loop controller usable as ``rollout(control=...)``."""

    policy: Any  # callable obs -> action
    observe: ObservationBuilder
    command: np.ndarray = field(default_factory=lambda: np.zeros(CMD_DIM, dtype=np.float32))
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(ACT_DIM, dtype=np.float32))
    log: list[dict[str, np.ndarray]] = field(default_factory=list)

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        obs = self.observe(data, self.last_action, self.command)
        action = self.policy(obs)
        self.last_action = action
        data.ctrl[:] = DEFAULT_POSE + action
        self.log.append({"obs": obs, "action": action})


def load_model_for_policy(scene: str | Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = PHYSICS_DT  # upstream infer_policy.py overrides 0.002 -> 0.005
    return model, mujoco.MjData(model)


# Upstream infer_policy.py:1376-1386 starts the trunk 0.125 m up, level, joints at
# DEFAULT_POSE, without settling onto the floor.
UPSTREAM_INIT_HEIGHT = 0.125
INIT_MODES = ("upstream", "settle")


def upstream_init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    observe = ObservationBuilder(model)
    data.qpos[0:3] = [0.0, 0.0, UPSTREAM_INIT_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[observe.qpos_idx] = DEFAULT_POSE
    data.qvel[:] = 0.0
    data.ctrl[:] = DEFAULT_POSE


def run(
    scene: str | Path,
    policy: Any,
    duration_s: float,
    twist: tuple[float, float, float] = (0.0, 0.0, 0.0),
    keyframe: str = "STAND",
    init: str = "upstream",
    recorder: Callable[[mujoco.MjModel, ControlFn], ControlFn] | None = None,
) -> tuple[sim_rollout.RolloutResult, PolicyController, dict[str, float]]:
    """Roll out ``policy`` with a constant twist command; return metrics.

    ``init="upstream"`` reproduces the upstream start state; ``"settle"`` lowers the
    keyframe pose onto the floor first. ``recorder(model, control)`` may wrap the
    control function (e.g. telemetry), receiving the policy command via closure.
    """
    if init not in INIT_MODES:
        raise ValueError(f"init must be one of {INIT_MODES}")
    model, data = load_model_for_policy(scene)
    command = np.zeros(CMD_DIM, dtype=np.float32)
    command[0:3] = twist
    ctrl = PolicyController(policy=policy, observe=ObservationBuilder(model), command=command)
    start = None

    def control(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal start
        if start is None:
            start = d.qpos[0:2].copy()
        ctrl(m, d)

    loop: ControlFn = control if recorder is None else recorder(model, control)
    result = sim_rollout.rollout(
        model,
        data,
        keyframe,
        duration_s,
        control_hz=CONTROL_HZ,
        control=loop,
        settle=init == "settle",
        init=upstream_init if init == "upstream" else None,
    )
    assert start is not None
    displacement = data.qpos[0:2] - start
    extra = {
        "displacement_x_m": float(displacement[0]),
        "displacement_y_m": float(displacement[1]),
        "mean_speed_x_mps": float(displacement[0] / duration_s),
        "max_abs_action": float(np.abs(np.stack([e["action"] for e in ctrl.log])).max()),
    }
    return result, ctrl, extra


# Scenario name -> (twist command, duration). Short, fixed, deterministic.
SCENARIOS: dict[str, tuple[tuple[float, float, float], float]] = {
    "stand_still": ((0.0, 0.0, 0.0), 10.0),
    "walk_forward_0.2": ((0.2, 0.0, 0.0), 10.0),
    "walk_backward_0.15": ((-0.15, 0.0, 0.0), 10.0),
    "turn_left_0.5": ((0.0, 0.0, 0.5), 10.0),
}


def evaluate(scene: str | Path, policy_path: str | Path, init: str = "upstream") -> dict[str, Any]:
    """Characterize a policy over SCENARIOS (labelled with the actuator model used)."""
    policy = OnnxPolicy(policy_path)
    report: dict[str, Any] = {
        "policy": Path(policy_path).name,
        "scene": Path(scene).name,
        "actuator": "mjcf-position (not BAM; known sim-to-sim gap)",
        "init": init,
        "physics_dt": PHYSICS_DT,
        "control_hz": CONTROL_HZ,
        "mujoco": mujoco.__version__,
        "scenarios": {},
    }
    for name, (twist, duration) in SCENARIOS.items():
        result, ctrl, extra = run(scene, policy, duration, twist=twist, init=init)
        yaw_rate = float(np.mean([e["obs"][2] for e in ctrl.log[len(ctrl.log) // 2 :]]))
        report["scenarios"][name] = {
            "command": twist,
            **result.summary(),
            **extra,
            "mean_gyro_z_second_half": yaw_rate,
        }
    return report
