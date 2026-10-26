"""BAM M6 model of the Dynamixel XL330 servo for CPU MuJoCo (backlog WP1-06).

Port of the parts of Rhoban/bam (https://github.com/Rhoban/bam, Apache-2.0,
commit 62bd8ce12154340be97e06f7f41a0ca8f116d967, branch mjlab_frictionloss) that
microduck_rl's ``scripts/infer_policy.py`` uses on CPU:

- ``bam/params/xl330/m6.json``                identified parameters (copied below)
- ``VoltageControlledActuator.compute_control``  firmware P-loop -> PWM duty -> volts
- ``VoltageControlledActuator.compute_torque``   DC motor with back-EMF
- ``Model.compute_frictions`` (m6)            load-dependent, directional,
                                                 Stribeck + quadratic friction budget
- ``mujoco.MujocoController.update``           per-physics-step torque + friction write
- ``infer_policy.load_mujoco_with_bam``        spec edits (motors, force range,
                                                 zero joint damping, stiff friction)

Ported rather than installed so no third-party code runs here (ADR-0010). The
numbers and equations follow the pinned source; tests compare against values
computed by hand from these formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

BAM_SOURCE = "Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967 (Apache-2.0)"

# bam/params/xl330/m6.json at the pinned commit.
M6 = {
    "kt": 0.36601349688984386,
    "R": 2.8113923539223227,
    "armature": 0.0018077432831600838,
    "friction_base": 0.004771183165566,
    "friction_stribeck": 0.004676345799486616,
    "load_friction_motor": 0.2667860954283698,
    "load_friction_external": 8.515871897059342e-06,
    "load_friction_motor_stribeck": 1.0722918395099123e-05,
    "load_friction_external_stribeck": 0.08077928978935671,
    "load_friction_motor_quad": 0.009972471242139415,
    "load_friction_external_quad": 0.004902565732332559,
    "dtheta_stribeck": 2.890372094130307,
    "alpha": 8.683259907618984,
    "friction_viscous": 0.005359668274599504,
}

# bam/dynamixel/actuator.py: XL330 firmware constants.
XL330_ENCODER_COUNTS_PER_REV = 4096
XL330_KP_DIVISOR = 256
XL330_PWM_LIMIT = 885
ERROR_GAIN = (XL330_ENCODER_COUNTS_PER_REV / (2 * np.pi)) / (XL330_KP_DIVISOR * XL330_PWM_LIMIT)
MAX_PWM = 1.0

# microduck_rl infer_policy.py defaults (mirror of the training actuator config).
KP_FW = 200.0
VIN = 7.4
VIN_DROP_GAIN = 0.1
VIN_MIN = 6.0
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


def force_limit(vin: float = VIN) -> float:
    return vin * M6["kt"] / M6["R"]


def compute_control(q_target: np.ndarray, q: np.ndarray, kp: float, vin: float) -> np.ndarray:
    """Firmware P-loop: duty = clip(error_gain * kp * (q_target - q)), volts = vin * duty."""
    duty = np.clip((q_target - q) * kp * ERROR_GAIN, -MAX_PWM, MAX_PWM)
    return np.asarray(vin * duty)


def compute_torque(volts: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """DC motor with back-EMF: tau = kt V / R - kt^2 dq / R."""
    kt, r = M6["kt"], M6["R"]
    return np.asarray(kt * volts / r - kt**2 * dq / r)


def compute_frictions(
    motor_torque: np.ndarray, external_torque: np.ndarray, dq: np.ndarray
) -> tuple[np.ndarray, float]:
    """M6 friction budget -> (frictionloss, damping) for MuJoCo's dofs."""
    p = M6
    gearbox = np.abs(
        external_torque * p["load_friction_external"] - motor_torque * p["load_friction_motor"]
    )
    gearbox_stribeck = np.abs(
        external_torque * p["load_friction_external_stribeck"]
        - motor_torque * p["load_friction_motor_stribeck"]
    )
    stribeck = np.exp(-(np.abs(dq / p["dtheta_stribeck"]) ** p["alpha"]))
    frictionloss = p["friction_base"] + gearbox
    frictionloss = frictionloss + stribeck * p["friction_stribeck"]
    frictionloss = frictionloss + gearbox_stribeck * stribeck
    opposing = np.sign(external_torque) != np.sign(motor_torque)
    motor_side = np.abs(external_torque) < np.abs(motor_torque)
    external_side = np.abs(external_torque) > np.abs(motor_torque)
    quad = (
        motor_side * p["load_friction_external_quad"] * np.abs(external_torque) ** 2
        + external_side * p["load_friction_motor_quad"] * np.abs(motor_torque) ** 2
    )
    frictionloss = frictionloss + stribeck * quad * opposing
    return np.asarray(frictionloss), p["friction_viscous"]


def load_model(scene: str | Path, timestep: float, vin: float = VIN) -> mujoco.MjModel:
    """Compile ``scene`` with every non-passive actuator turned into a BAM-driven motor."""
    spec = mujoco.MjSpec.from_file(str(scene))
    limit = force_limit(vin)
    joints = {j.name: j for j in spec.joints}
    for act in spec.actuators:
        target = act.target
        if target.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-limit, limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        joint = joints.get(target)
        if joint is not None:
            joint.damping = np.zeros((3, 1))
            joint.frictionloss = 0.0
            joint.solref_friction = STIFF_SOLREF_FRICTION
            joint.solimp_friction = STIFF_SOLIMP_FRICTION
    model = spec.compile()
    model.opt.timestep = timestep
    return model


@dataclass
class Controller:
    """Per-physics-step BAM update (port of ``bam.mujoco.MujocoController``).

    Set ``q_target`` from the policy; call ``update(data)`` before every
    ``mj_step``. Writes torque into ``data.ctrl`` and the friction budget into
    ``model.dof_frictionloss`` / ``model.dof_damping``.
    """

    model: mujoco.MjModel
    kp: float = KP_FW
    vin: float = VIN
    vin_drop_gain: float | None = VIN_DROP_GAIN
    vin_min: float | None = VIN_MIN
    act_ids: np.ndarray = field(init=False)
    joint_ids: np.ndarray = field(init=False)
    qpos_idx: np.ndarray = field(init=False)
    dof_idx: np.ndarray = field(init=False)
    q_target: np.ndarray = field(init=False)
    last_torque: np.ndarray = field(init=False)
    last_vin: float = field(init=False)

    def __post_init__(self) -> None:
        m = self.model
        names = [m.actuator(i).name for i in range(m.nu)]
        self.act_ids = np.array(
            [
                i
                for i, n in enumerate(names)
                if not m.joint(m.actuator_trnid[i, 0]).name.startswith("passive_")
            ]
        )
        self.joint_ids = m.actuator_trnid[self.act_ids, 0]
        self.qpos_idx = m.jnt_qposadr[self.joint_ids].astype(int)
        self.dof_idx = m.jnt_dofadr[self.joint_ids].astype(int)
        m.dof_armature[self.dof_idx] = M6["armature"]
        self.q_target = np.zeros(len(self.act_ids))
        self.last_torque = np.zeros(len(self.act_ids))
        self.last_vin = self.vin

    def reset(self, data: mujoco.MjData) -> None:
        """Call after the start state is set (upstream: ``bam_ctrl.reset(data.qpos)``)."""
        mujoco.mj_setConst(self.model, data)
        self.q_target = data.qpos[self.qpos_idx].copy()
        self.last_torque[:] = 0.0

    def update(self, data: mujoco.MjData) -> None:
        q = data.qpos[self.qpos_idx]
        dq = data.qvel[self.dof_idx]
        vin = self.vin
        if self.vin_drop_gain is not None:
            vin = vin - self.vin_drop_gain * float(np.abs(self.last_torque).sum())
            if self.vin_min is not None:
                vin = max(vin, self.vin_min)
        self.last_vin = vin
        torque = compute_torque(compute_control(self.q_target, q, self.kp, vin), dq)
        self.last_torque = torque.copy()
        data.ctrl[self.act_ids] = torque

        external = -data.qfrc_bias[self.dof_idx] + data.qfrc_constraint[self.dof_idx]
        friction_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        friction = np.zeros(len(self.dof_idx))
        if friction_rows.any():
            # Upstream matches efc_id (a dof index for friction rows) against the
            # *joint* index; with a free root these differ by 5. Kept as upstream
            # does it, since the goal is to reproduce upstream's CPU simulator.
            ids, forces = data.efc_id[friction_rows], data.efc_force[friction_rows]
            for k, joint in enumerate(self.joint_ids):
                friction[k] = forces[ids == joint].sum()
        external = external - friction
        frictionloss, damping = compute_frictions(data.qfrc_actuator[self.dof_idx], external, dq)
        self.model.dof_frictionloss[self.dof_idx] = frictionloss
        self.model.dof_damping[self.dof_idx] = damping
