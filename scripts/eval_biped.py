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
from bipedlab.rl import biped_velocity_env as bve

SETTLE_S = 2.0
TWISTS = [(0.15, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.8)]

# Robustness cases (G4.3): one factor each, DR off, forward command.
ROBUST_TWIST = TWISTS[0]
ROBUST_FRICTION = (0.5, 1.5)  # foot friction outside the DR range (0.7, 1.3)
PUSH_AT_S = 4.0
PUSH_DV = 0.3  # m/s in the trunk's heading frame, 1.5x the DR push
ROBUST_PUSHES = {"push_forward": (PUSH_DV, 0.0), "push_sideways": (0.0, PUSH_DV)}
ROBUST_DELAYS = (1, 2, 3)  # control periods of 20 ms


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
) -> dict[str, Any]:
    env = bve.BipedVelocityEnv(
        num_envs=num_envs,
        seed=seed,
        episode_s=seconds + 1.0,
        common_step=2000 * 24,
        dr=dr,
        action_delay=action_delay,
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
    knee: list[np.ndarray] = []
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
        knee.append(out.joint_torque[alive][:, bve.KNEES].ravel())
        for k, v in out.terms.items():
            terms.setdefault(k, []).append(float(v[alive].mean()))
    vx_arr = np.array(vx) if vx else np.full((1, num_envs), np.nan)
    k = np.concatenate(knee) if knee else np.zeros(1)
    return {
        "policy": "zero-action" if policy is None else "onnx",
        "case": case,
        "action_delay_ms": action_delay * bve.STEP_DT * 1000.0,
        "foot_friction": foot_friction,
        "push_dv": None if push_dv is None else list(push_dv),
        "twist": list(twist),
        "dr": dr,
        "num_envs": num_envs,
        "seconds": seconds,
        "falls": int((~alive).sum()),
        "vx_mean": float(np.nanmean(vx_arr)) if np.isfinite(vx_arr).any() else None,
        "knee_torque": {
            "mean": float(k.mean()),
            "p95": float(np.percentile(k, 95)),
            "max": float(k.max()),
        },  # fmt: skip
        "slew_limited_fraction": float(np.concatenate(limited).mean()),
        # Without delay: |policy target - previous target|; delayed cases mix two periods.
        "wanted_target_step_mean": (
            float(np.concatenate(wanted_step).mean()) if action_delay == 0 else None
        ),
        "terms_per_step": {name: float(np.mean(v)) for name, v in terms.items()},
    }


def robustness_cases(
    policy: BatchOnnxPolicy | None, num_envs: int, seconds: float, seed: int
) -> list[dict[str, Any]]:
    """G4.3 robustness: friction, pushes and latency, one factor per case, DR off."""
    run = partial(run_case, policy, ROBUST_TWIST, num_envs, seconds, seed, False)
    cases = [run(case=f"friction_{mu}", foot_friction=mu) for mu in ROBUST_FRICTION]
    cases += [run(case=name, push_dv=dv) for name, dv in ROBUST_PUSHES.items()]
    cases += [
        run(case=f"latency_{round(k * bve.STEP_DT * 1000)}ms", action_delay=k)
        for k in ROBUST_DELAYS
    ]
    return cases


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
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    policy = BatchOnnxPolicy(args.policy)
    cases = []
    for dr in [False, True] if args.dr else [False]:
        for twist in TWISTS:
            cases.append(run_case(policy, twist, args.envs, args.seconds, args.seed, dr))
        cases.append(run_case(None, TWISTS[0], args.envs, args.seconds, args.seed, dr))
    if args.robustness:
        cases.extend(robustness_cases(policy, args.envs, args.seconds, args.seed))
    report = {"environment": bench.environment(), "policy": str(args.policy), "cases": cases}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
