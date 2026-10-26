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
                 (optional robot-runtime stage: see PostProcess)
    timing: physics dt 0.005 s, decimation 4 -> 50 Hz

Joint order is the MJCF actuator order (left leg 0-4, neck/head 5-8, right leg 9-13).

Actuators: training uses the BAM XL330 model. ``actuator="mjcf"`` drives MuJoCo's
position actuators (upstream's ``--no-bam`` path, a known sim-to-sim gap);
``actuator="bam"`` uses our port of BAM M6 (upstream's default CPU path, see
``bam_xl330``). Every report names the actuator it used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bipedlab.sim import bam_xl330
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

HEAD_JOINTS = slice(5, 9)  # neck_pitch, head_pitch, head_yaw, head_roll
LEG_JOINTS = np.r_[0:5, 9:14]


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, xyz = quat_wxyz[0], quat_wxyz[1:4]
    t = np.cross(xyz, vec) * 2.0
    return np.asarray(vec - w * t + np.cross(xyz, t))


class PolicyError(RuntimeError):
    pass


@dataclass
class ObservationBuilder:
    model: mujoco.MjModel
    # Control steps of lag on joint_vel. Training uses exactly 1
    # (microduck_velocity_env_cfg.py: Dynamixel velocity is ~1 period old);
    # upstream's CPU simulator uses 0.
    joint_vel_lag: int = 0
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
        self._vel_history: list[np.ndarray] = []

    def __call__(
        self, data: mujoco.MjData, last_action: np.ndarray, command: np.ndarray
    ) -> np.ndarray:
        gravity = quat_rotate_inverse(data.xquat[self.trunk_id], np.array([0.0, 0.0, -1.0]))
        joint_vel = data.qvel[self.qvel_idx].copy()
        if self.joint_vel_lag:
            self._vel_history.append(joint_vel)
            if len(self._vel_history) > self.joint_vel_lag + 1:
                self._vel_history.pop(0)
            joint_vel = self._vel_history[0]
        obs = np.concatenate(
            [
                data.sensordata[self.gyro_adr : self.gyro_adr + 3],
                gravity,
                data.qpos[self.qpos_idx] - DEFAULT_POSE,
                joint_vel,
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


@dataclass(frozen=True)
class PostProcess:
    """Optional model of the robot runtime's action stage (microduck ``robotd``).

    Reimplements ``robotd/src/control.rs`` at the pinned sha: ``target = DEFAULT_POSE +
    scale * action``, then a first-order low-pass ``alpha * target + (1 - alpha) *
    previous`` on head and leg targets (skipped on the first tick). ``scale`` is
    ``action_scale * nominal_voltage / clamp(supply_voltage, 6.0, 9.5)`` when a supply
    voltage is given. The observation's ``last_action`` stays the raw policy output, as
    in robotd. ``None`` alphas mean pass-through. Upstream's own sim (infer_policy.py)
    uses none of this, so the default instance is the identity.
    """

    action_scale: float = 1.0
    head_lowpass: float | None = None
    legs_lowpass: float | None = None
    supply_voltage: float | None = None
    nominal_voltage: float = 7.4

    def __post_init__(self) -> None:
        for name in ("head_lowpass", "legs_lowpass"):
            alpha = getattr(self, name)
            if alpha is not None and not 0.0 < alpha <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {alpha}")

    @property
    def scale(self) -> float:
        if self.supply_voltage is None:
            return self.action_scale
        return self.action_scale * self.nominal_voltage / min(max(self.supply_voltage, 6.0), 9.5)

    @property
    def is_identity(self) -> bool:
        return self.scale == 1.0 and self.head_lowpass is None and self.legs_lowpass is None

    def targets(self, action: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
        target = (DEFAULT_POSE + self.scale * action).astype(np.float32)
        if previous is not None:
            if self.head_lowpass is not None:
                a = self.head_lowpass
                target[HEAD_JOINTS] = a * target[HEAD_JOINTS] + (1 - a) * previous[HEAD_JOINTS]
            if self.legs_lowpass is not None:
                a = self.legs_lowpass
                target[LEG_JOINTS] = a * target[LEG_JOINTS] + (1 - a) * previous[LEG_JOINTS]
        return target

    def describe(self) -> dict[str, float | None]:
        return {
            "action_scale": self.action_scale,
            "effective_scale": self.scale,
            "head_lowpass": self.head_lowpass,
            "legs_lowpass": self.legs_lowpass,
            "supply_voltage": self.supply_voltage,
        }


# robotd walking defaults (deploy/robotd.toml): scale 0.9, low-pass 0.5 head / 0.7 legs.
# The voltage term is left off: with a 7.4 V supply it is exactly 1.0.
ROBOTD_WALKING = PostProcess(action_scale=0.9, head_lowpass=0.5, legs_lowpass=0.7)
POSTPROCESS_PRESETS = {"none": PostProcess(), "robotd-walking": ROBOTD_WALKING}


@dataclass
class PolicyController:
    """Closed-loop controller usable as ``rollout(control=...)``."""

    policy: Any  # callable obs -> action
    observe: ObservationBuilder
    command: np.ndarray = field(default_factory=lambda: np.zeros(CMD_DIM, dtype=np.float32))
    last_action: np.ndarray = field(default_factory=lambda: np.zeros(ACT_DIM, dtype=np.float32))
    post: PostProcess = field(default_factory=PostProcess)
    bam: bam_xl330.Controller | None = None
    log: list[dict[str, np.ndarray]] = field(default_factory=list)
    _previous: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def last_target(self) -> np.ndarray:
        return DEFAULT_POSE.copy() if self._previous is None else self._previous

    def __call__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        obs = self.observe(data, self.last_action, self.command)
        action = self.policy(obs)
        self.last_action = action
        target = self.post.targets(action, self._previous)
        self._previous = target
        if self.bam is None:
            data.ctrl[:] = target
        else:
            self.bam.q_target[:] = target
        self.log.append({"obs": obs, "action": action})


@dataclass(frozen=True)
class TrainingMatch:
    """Training-time effects upstream's CPU simulator leaves out (backlog WP1-11).

    Defaults reproduce upstream's simulator. Training values, from
    microduck_constants.py / microduck_velocity_env_cfg.py at the pinned sha:
    BAM command delay uniform 3..6 physics steps, supply 6.5..8.2 V, and a
    1-control-step lag on the joint_vel observation.
    """

    bam_delay_steps: int = 0
    vin: float = bam_xl330.VIN
    joint_vel_lag: int = 0

    def describe(self) -> dict[str, float]:
        return {
            "bam_delay_steps": self.bam_delay_steps,
            "vin": self.vin,
            "joint_vel_lag": self.joint_vel_lag,
        }


ACTUATORS = ("mjcf", "bam")
ACTUATOR_LABELS = {
    "mjcf": "mjcf-position (not BAM; known sim-to-sim gap)",
    "bam": f"bam-xl330-m6 (CPU port of {bam_xl330.BAM_SOURCE}; drop gain 0.1)",
}


def load_model_for_policy(
    scene: str | Path, actuator: str = "mjcf", vin: float = bam_xl330.VIN
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load ``scene`` at the policy timestep with MJCF position actuators or BAM motors."""
    if actuator not in ACTUATORS:
        raise ValueError(f"actuator must be one of {ACTUATORS}")
    if actuator == "bam":
        model = bam_xl330.load_model(scene, PHYSICS_DT, vin=vin)
    else:
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
    keyframe: str | None = "STAND",
    init: str = "upstream",
    recorder: Callable[[mujoco.MjModel, ControlFn], ControlFn] | None = None,
    post: PostProcess | None = None,
    command_fn: Callable[[float], np.ndarray] | None = None,
    after_init: ControlFn | None = None,
    actuator: str = "mjcf",
    match: TrainingMatch | None = None,
) -> tuple[sim_rollout.RolloutResult, PolicyController, dict[str, float]]:
    """Roll out ``policy`` with a constant twist command; return metrics.

    ``init="upstream"`` reproduces the upstream start state; ``"settle"`` lowers the
    keyframe pose onto the floor first. ``recorder(model, control)`` may wrap the
    control function (e.g. telemetry), receiving the policy command via closure.
    ``post`` optionally models the robot runtime's action stage (see PostProcess).
    ``command_fn(t)`` replaces the constant twist with a full 13-D command as a
    function of seconds since the first control step (phase-encoded behaviours).
    ``after_init`` edits the state after the start pose is set (e.g. place a ball).
    ``actuator="bam"`` drives BAM XL330 motors (torque recomputed every physics step).
    ``match`` adds training-time effects (see TrainingMatch); delay and vin need BAM.
    """
    if init not in INIT_MODES:
        raise ValueError(f"init must be one of {INIT_MODES}")
    match = match or TrainingMatch()
    if actuator != "bam" and (match.bam_delay_steps or match.vin != bam_xl330.VIN):
        raise ValueError("bam_delay_steps and vin apply to actuator='bam' only")
    model, data = load_model_for_policy(scene, actuator, vin=match.vin)
    bam = (
        bam_xl330.Controller(model, vin=match.vin, delay_steps=match.bam_delay_steps)
        if actuator == "bam"
        else None
    )
    command = np.zeros(CMD_DIM, dtype=np.float32)
    command[0:3] = twist
    ctrl = PolicyController(
        policy=policy,
        observe=ObservationBuilder(model, joint_vel_lag=match.joint_vel_lag),
        command=command,
        post=post or PostProcess(),
        bam=bam,
    )
    start = None
    t0 = 0.0

    def control(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        nonlocal start, t0
        if start is None:
            start, t0 = d.qpos[0:2].copy(), d.time
            if bam is not None:
                bam.reset(d)  # after init/settle, as upstream resets after its init
        if command_fn is not None:
            ctrl.command[:] = command_fn(d.time - t0)
        ctrl(m, d)

    def init_fn(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        if init == "upstream":
            upstream_init(m, d)
        if after_init is not None:
            after_init(m, d)

    loop: ControlFn = control if recorder is None else recorder(model, control)
    from bipedlab.telemetry.recorder import Recorder

    if isinstance(loop, Recorder) and loop.target is None:
        loop.target = lambda: ctrl.last_target
    result = sim_rollout.rollout(
        model,
        data,
        keyframe,
        duration_s,
        control_hz=CONTROL_HZ,
        control=loop,
        settle=init == "settle",
        init=init_fn if init == "upstream" or after_init is not None else None,
        substep=None if bam is None else (lambda m, d: bam.update(d)),
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


# Start-state perturbation for repeat runs: below one XL330 encoder count
# (2 pi / 4096 = 1.5e-3 rad), so every repeat is a start the real robot cannot tell apart.
REPEAT_PERTURB_RAD = 1.0e-3


def perturb_joints(seed: int, amplitude_rad: float = REPEAT_PERTURB_RAD) -> ControlFn:
    """``after_init`` hook: add seeded uniform noise to the actuated joint positions."""

    def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        idx = ObservationBuilder(model).qpos_idx
        rng = np.random.default_rng(seed)
        data.qpos[idx] += rng.uniform(-amplitude_rad, amplitude_rad, len(idx))

    return init


def _spread(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def evaluate(
    scene: str | Path,
    policy_path: str | Path,
    init: str = "upstream",
    post: PostProcess | None = None,
    actuator: str = "mjcf",
    match: TrainingMatch | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    """Characterize a policy over SCENARIOS (labelled with the actuator model used).

    ``repeats > 1`` adds runs from perturbed starts (``perturb_joints`` with seeds
    1..repeats-1) and reports the spread of the tracking metrics. The BAM gait is
    sensitive enough that last-bit differences between CPUs change a single run's
    numbers by several percent, so a single run is one sample, not a measurement.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    match = match or TrainingMatch()
    policy = OnnxPolicy(policy_path)
    post = post or PostProcess()
    report: dict[str, Any] = {
        "policy": Path(policy_path).name,
        "scene": Path(scene).name,
        "actuator": ACTUATOR_LABELS[actuator],
        "init": init,
        "postprocess": post.describe(),
        "training_match": match.describe(),
        "physics_dt": PHYSICS_DT,
        "control_hz": CONTROL_HZ,
        "mujoco": mujoco.__version__,
        "repeats": repeats,
        "scenarios": {},
    }
    for name, (twist, duration) in SCENARIOS.items():
        samples: dict[str, list[float]] = {"vx_mps": [], "vy_mps": [], "yaw_rate": []}
        falls = 0
        for k in range(repeats):
            result, ctrl, extra = run(
                scene,
                policy,
                duration,
                twist=twist,
                init=init,
                post=post,
                actuator=actuator,
                match=match,
                after_init=perturb_joints(k) if k else None,
            )
            yaw_rate = float(np.mean([e["obs"][2] for e in ctrl.log[len(ctrl.log) // 2 :]]))
            if k == 0:
                report["scenarios"][name] = {
                    "command": twist,
                    **result.summary(),
                    **extra,
                    "mean_gyro_z_second_half": yaw_rate,
                }
            samples["vx_mps"].append(extra["mean_speed_x_mps"])
            samples["vy_mps"].append(extra["displacement_y_m"] / duration)
            samples["yaw_rate"].append(yaw_rate)
            falls += int(result.fell)
        if repeats > 1:
            report["scenarios"][name]["spread"] = {
                "falls": falls,
                **{key: _spread(v) for key, v in samples.items()},
            }
    return report
