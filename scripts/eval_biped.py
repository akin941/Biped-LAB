"""G4.3: score an ONNX policy for our own biped_v1 in ``BipedVelocityEnv``.

Runs the policy (39 -> 10 contract, see ``bipedlab.rl.biped_velocity_env``) with frozen
commands and reports, per case, the body-frame forward speed after a 2 s settle,
falls, the per-step reward terms and the knee torque (mean, 95th percentile and max
of |actuator force|), and how often the 10 rad/s target slew limit of the runtime
supervisor cut a step (``slew_limited_fraction``: env-steps with at least one joint
limited, WP4-11) and the mean wanted target step per joint (rad). ``--dr`` adds
the same cases with domain randomization on (friction, mass, CoM, armature, IMU
tilt, encoder bias, random pushes). A zero-action run is the baseline.

``--robustness`` adds the G4.3 robustness cases one factor at a time, at the 0.15 m/s
command with DR off, so a failure points at its cause (WP4-12): foot friction fixed
below and above the DR range, a single forward and a single sideways push of
``PUSH_DV`` at ``PUSH_AT_S``, and 1-3 control periods (20-60 ms) of action latency.
``--terrain`` adds the same command on a floor of random-height tiles instead of the flat
plane, ``TERRAIN_HEIGHTS`` peak to peak (WP4-31; training used the flat floor).

Each case also reports per joint type (both legs pooled) the |torque| RMS / p95 / max,
the |joint speed| p95 / max and the mean / p95 mechanical power |torque * speed|
(``joints``, WP4-25), and the report's ``sizing`` block turns the walking cases into
requirements with the G4.1 margin (2x the worst max torque and p95 speed), so WP4-01
can be re-done from our own learned gait instead of Microduck's dynamic factors.

The knee numbers re-check the WP4-01 requirement with a learned gait: the committed
model uses the XL330-class placeholder (0.52 N*m force limit), so a p95 close to
0.52 means the knee is saturating.

Usage: python scripts/eval_biped.py runs/biped/policy.onnx --dr --out reports/g4-3.json
"""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab import bench
from bipedlab.design import biped
from bipedlab.rl import biped_velocity_env as bve

SCENES = {
    "biped_v1": bve.DEFAULT_SCENE,
    "biped_ds3218": bve.DS3218_SCENE,
    "biped_ds3218b": bve.DS3218B_SCENE,
    "biped_ds3218b_heavy": bve.DS3218B_HEAVY_SCENE,
}
SETTLE_S = 2.0
TWISTS = [(0.15, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.8)]

# Robustness cases (G4.3): one factor each, DR off, forward command.
ROBUST_TWIST = TWISTS[0]
# Actor observation (WP4-37): "train" = the training noise, gyro/gravity lag and
# one-period-old joint velocities; "no-vel-lag" = the same with current joint
# velocities; "clean" = no noise or lag, current joint velocities (the runtime before
# WP4-37, MAC-45's "runtime"); "runtime" = no noise, one-period-old joint velocities
# (runtime/src/policy.rs since WP4-37).
OBS_MODES = ("train", "no-vel-lag", "clean", "runtime")
OBS_ENV_KWARGS: dict[str, dict[str, Any]] = {
    "train": {"noise": True, "joint_vel_lag": None},
    "no-vel-lag": {"noise": True, "joint_vel_lag": False},
    "clean": {"noise": False, "joint_vel_lag": None},
    "runtime": {"noise": False, "joint_vel_lag": True},
}
ROBUST_FRICTION = (0.5, 1.5)  # foot friction outside the DR range (0.7, 1.3)
PUSH_AT_S = 4.0
PUSH_DV = 0.3  # m/s in the trunk's heading frame, 1.5x the DR push
ROBUST_PUSHES = {"push_forward": (PUSH_DV, 0.0), "push_sideways": (0.0, PUSH_DV)}
ROBUST_DELAYS = (1, 2, 3)  # control periods of 20 ms
# Uneven floor (WP4-31): peak-to-peak heights in m; biped_v1 was trained on a flat plane.
TERRAIN_HEIGHTS = (0.005, 0.01, 0.02)
SIZING_MARGIN = 2.0  # G4.1
JOINT_GROUPS = {
    j: np.array([bve.JOINT_INDEX[f"{s}_{j}"] for s in biped.SIDES]) for j in biped.LEG_JOINTS
}


class BatchOnnxPolicy:
    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
        if inp.shape[-1] != bve.OBS_DIM or out.shape[-1] != bve.ACT_DIM:
            raise SystemExit(f"policy is {inp.shape}->{out.shape}, contract 39->10")
        self.input_name, self.output_name = inp.name, out.name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        rows = [self.session.run([self.output_name], {self.input_name: o[None]})[0] for o in obs]
        actions = np.concatenate(rows).astype(np.float32)
        if not np.isfinite(actions).all():
            raise SystemExit("policy produced a non-finite action")
        return actions


def push_body_frame(d: Any, trunk_id: int, dv: tuple[float, float]) -> None:
    """Add a horizontal velocity kick given along the trunk's heading (x) and left (y)."""
    w, x, y, z = d.xquat[trunk_id]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    d.qvel[0] += c * dv[0] - s * dv[1]
    d.qvel[1] += s * dv[0] + c * dv[1]


def run_case(
    policy: BatchOnnxPolicy | None,
    twist: tuple[float, float, float],
    num_envs: int,
    seconds: float,
    seed: int,
    dr: bool,
    *,
    case: str = "nominal",
    action_delay: int = 0,
    foot_friction: float | None = None,
    push_dv: tuple[float, float] | None = None,
    torque_limit: float | None = None,
    motor_no_load_speed: float | None = None,
    rough_floor: float = 0.0,
    obs_mode: str = "train",
    scene: str | Path = bve.DEFAULT_SCENE,
    joint_feedback: str = "measured",
    servo_dr: bool = False,
) -> dict[str, Any]:
    if obs_mode not in OBS_MODES:
        raise ValueError(f"obs_mode must be one of {OBS_MODES}")
    env = bve.BipedVelocityEnv(
        **OBS_ENV_KWARGS[obs_mode],
        scene=scene,
        joint_feedback=joint_feedback,
        servo_dr=servo_dr,
        num_envs=num_envs,
        seed=seed,
        episode_s=seconds + 1.0,
        common_step=2000 * 24,
        dr=dr,
        action_delay=action_delay,
        torque_limit=torque_limit,
        motor_no_load_speed=motor_no_load_speed,
        rough_floor=rough_floor,
    )
    if foot_friction is not None:
        if dr:
            raise ValueError("a fixed foot friction case needs dr=False")
        env.model.geom_friction[env.foot_geoms, 0] = foot_friction
    env.freeze_commands(np.array(twist))
    push_step = round(PUSH_AT_S / bve.STEP_DT)
    steps, settle = round(seconds / bve.STEP_DT), round(SETTLE_S / bve.STEP_DT)
    alive = np.ones(num_envs, dtype=bool)
    vx: list[np.ndarray] = []
    vy: list[np.ndarray] = []
    wz: list[np.ndarray] = []
    knee: list[np.ndarray] = []
    torque_rows: list[np.ndarray] = []
    speed_rows: list[np.ndarray] = []
    limited: list[np.ndarray] = []
    wanted_step: list[np.ndarray] = []
    terms: dict[str, list[float]] = {}
    obs = env.obs
    for t in range(steps):
        actions = np.zeros((num_envs, bve.ACT_DIM), np.float32) if policy is None else policy(obs)
        before = env.last_target.copy()
        out = env.step(actions)
        wanted = env.default_pose + bve.ACTION_SCALE * actions
        limited.append(out.slew_limited[alive])
        wanted_step.append(np.abs(wanted - before)[alive].ravel())
        alive &= ~out.terminated
        obs = out.obs
        if push_dv is not None and t == push_step:
            for i in np.flatnonzero(alive):
                push_body_frame(env.datas[int(i)], env.trunk_id, push_dv)
        if not alive.any():
            break
        if t >= settle:
            vx.append(np.where(alive, out.critic_obs[:, bve.OBS_DIM], np.nan))
            vy.append(np.where(alive, out.critic_obs[:, bve.OBS_DIM + 1], np.nan))
            # Critic's clean gyro (first actor term): body yaw rate. WP4-21: MAC-16
            # saw -0.15..-0.19 rad/s yaw drift at a zero yaw command in the runtime.
            wz.append(np.where(alive, out.critic_obs[:, 2], np.nan))
        knee.append(out.joint_torque[alive][:, bve.KNEES].ravel())
        if t >= settle:
            idx = np.flatnonzero(alive)
            torque_rows.append(out.joint_torque[idx])
            speed_rows.append(np.abs(np.array([env.datas[int(i)].qvel[env.qvel_idx] for i in idx])))
        for k, v in out.terms.items():
            terms.setdefault(k, []).append(float(v[alive].mean()))
    vx_arr = np.array(vx) if vx else np.full((1, num_envs), np.nan)

    def nanmean(rows: list[np.ndarray]) -> float | None:
        a = np.array(rows) if rows else np.full((1, num_envs), np.nan)
        return float(np.nanmean(a)) if np.isfinite(a).any() else None

    k = np.concatenate(knee) if knee else np.zeros(1)
    return {
        "policy": "zero-action" if policy is None else "onnx",
        "case": case,
        "action_delay_ms": action_delay * bve.STEP_DT * 1000.0,
        "foot_friction": foot_friction,
        "push_dv": None if push_dv is None else list(push_dv),
        "rough_floor": rough_floor,
        "obs_mode": obs_mode,
        "twist": list(twist),
        "dr": dr,
        "torque_limit": float(env.stall_torque[0]),
        "motor_no_load_speed": motor_no_load_speed,
        "num_envs": num_envs,
        "seconds": seconds,
        "falls": int((~alive).sum()),
        "vx_mean": float(np.nanmean(vx_arr)) if np.isfinite(vx_arr).any() else None,
        "vy_mean": nanmean(vy),
        "wz_mean": nanmean(wz),
        "knee_torque": {
            "mean": float(k.mean()),
            "p95": float(np.percentile(k, 95)),
            "max": float(k.max()),
        },  # fmt: skip
        "joints": joint_stats(torque_rows, speed_rows),
        "slew_limited_fraction": float(np.concatenate(limited).mean()),
        # Without delay: |policy target - previous target|; delayed cases mix two periods.
        "wanted_target_step_mean": (
            float(np.concatenate(wanted_step).mean()) if action_delay == 0 else None
        ),
        "terms_per_step": {name: float(np.mean(v)) for name, v in terms.items()},
    }


def joint_stats(
    torque_rows: list[np.ndarray], speed_rows: list[np.ndarray]
) -> dict[str, dict[str, float]] | None:
    """Per joint type (both legs pooled): |torque|, |speed| and |torque * speed| stats."""
    if not torque_rows:
        return None
    tau, omega = np.concatenate(torque_rows), np.concatenate(speed_rows)
    out = {}
    for name, cols in JOINT_GROUPS.items():
        t, w = tau[:, cols].ravel(), omega[:, cols].ravel()
        p = t * w
        out[name] = {
            "torque_rms": float(np.sqrt(np.mean(t**2))),
            "torque_p95": float(np.percentile(t, 95)),
            "torque_max": float(t.max()),
            "speed_p95": float(np.percentile(w, 95)),
            "speed_max": float(w.max()),
            "power_mean_w": float(p.mean()),
            "power_p95_w": float(np.percentile(p, 95)),
        }
    return out


def sizing_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """G4.1 requirements from the learned gait: 2x the worst walking max torque / p95 speed.

    Uses the policy's nominal cases (DR off and on) with a nonzero command; the zero-action
    baseline and the one-factor robustness cases are left out.
    """
    walking = [
        c for c in cases
        if c["policy"] == "onnx" and c["case"] == "nominal" and any(c["twist"])
        and c["joints"] is not None
    ]  # fmt: skip
    out: dict[str, Any] = {"margin": SIZING_MARGIN, "cases_used": len(walking), "joints": {}}
    limits = {c["torque_limit"] for c in walking if c.get("torque_limit") is not None}
    limit = max(limits) if limits else None
    for name in JOINT_GROUPS:
        rows = [c["joints"][name] for c in walking]
        if not rows:
            continue
        out["joints"][name] = {
            "torque_max": max(r["torque_max"] for r in rows),
            "torque_rms": max(r["torque_rms"] for r in rows),
            "speed_p95": max(r["speed_p95"] for r in rows),
            "required_torque": SIZING_MARGIN * max(r["torque_max"] for r in rows),
            "required_speed": SIZING_MARGIN * max(r["speed_p95"] for r in rows),
            # A walking max at the force limit means the demand was clipped (MAC-35):
            # the requirement is then a lower bound.
            "at_torque_limit": limit is not None
            and max(r["torque_max"] for r in rows) >= 0.99 * limit,
        }
    return out


def robustness_cases(
    policy: BatchOnnxPolicy | None,
    num_envs: int,
    seconds: float,
    seed: int,
    **env_kw: Any,
) -> list[dict[str, Any]]:
    """G4.3 robustness: friction, pushes and latency, one factor per case, DR off.

    ``env_kw`` are the shared ``run_case`` options (torque limit, motor curve,
    observation mode, scene, joint feedback)."""
    run = partial(run_case, policy, ROBUST_TWIST, num_envs, seconds, seed, False, **env_kw)
    cases = [run(case=f"friction_{mu}", foot_friction=mu) for mu in ROBUST_FRICTION]
    cases += [run(case=name, push_dv=dv) for name, dv in ROBUST_PUSHES.items()]
    cases += [
        run(case=f"latency_{round(k * bve.STEP_DT * 1000)}ms", action_delay=k)
        for k in ROBUST_DELAYS
    ]
    return cases


def terrain_cases(
    policy: BatchOnnxPolicy | None,
    num_envs: int,
    seconds: float,
    seed: int,
    **env_kw: Any,
) -> list[dict[str, Any]]:
    """WP4-31: the robustness command on random-height tile floors, DR off."""
    return [
        run_case(
            policy,
            ROBUST_TWIST,
            num_envs,
            seconds,
            seed,
            False,
            case=f"rough_{round(h * 1000)}mm",
            rough_floor=h,
            **env_kw,
        )  # fmt: skip
        for h in TERRAIN_HEIGHTS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", type=Path, help="path to a 39->10 .onnx policy")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dr", action="store_true", help="also run every case with DR on")
    parser.add_argument(
        "--robustness", action="store_true", help="add friction, push and latency cases"
    )
    parser.add_argument("--terrain", action="store_true", help="add uneven-floor cases (WP4-31)")
    parser.add_argument(
        "--torque-limit", type=float, default=None,
        help="actuator force limit in N*m (default: the model's 0.52; WP4-26)",
    )  # fmt: skip
    parser.add_argument(
        "--motor-no-load-speed", type=float, default=None,
        help="DC-motor no-load speed in rad/s: torque-speed curve on (WP4-28)",
    )  # fmt: skip
    parser.add_argument(
        "--obs-mode", choices=OBS_MODES, default="train",
        help="actor observation: training noise (default) or the runtime's (WP4-37)",
    )  # fmt: skip
    parser.add_argument(
        "--model", choices=sorted(SCENES), default="biped_v1",
        help="robot model: biped_v1 (bus servos) or biped_ds3218 (DS3218MG build, G4.5)",
    )  # fmt: skip
    parser.add_argument(
        "--joint-feedback", choices=bve.JOINT_FEEDBACK, default="measured",
        help="actor joint terms: measured (bus servos) or commanded targets (PWM, G4.5)",
    )  # fmt: skip
    parser.add_argument(
        "--servo-dr", action="store_true",
        help="every case on PWM servos with offset / deadband / stiffness draws (WP4-41)",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    tl = args.torque_limit
    lim = {
        "torque_limit": tl,
        "motor_no_load_speed": args.motor_no_load_speed,
        "obs_mode": args.obs_mode,
        "scene": SCENES[args.model],
        "joint_feedback": args.joint_feedback,
        "servo_dr": args.servo_dr,
    }
    policy = BatchOnnxPolicy(args.policy)
    cases = []
    for dr in [False, True] if args.dr else [False]:
        for twist in TWISTS:
            cases.append(run_case(policy, twist, args.envs, args.seconds, args.seed, dr, **lim))
        cases.append(run_case(None, TWISTS[0], args.envs, args.seconds, args.seed, dr, **lim))
    if args.robustness:
        cases.extend(robustness_cases(policy, args.envs, args.seconds, args.seed, **lim))
    if args.terrain:
        cases.extend(terrain_cases(policy, args.envs, args.seconds, args.seed, **lim))
    report = {
        "environment": bench.environment(),
        "policy": str(args.policy),
        "model": args.model,
        "joint_feedback": args.joint_feedback,
        "servo_dr": args.servo_dr,
        "cases": cases,
        "sizing": sizing_summary(cases),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
