"""G4.3: score an ONNX policy for our own biped_v1 in ``BipedVelocityEnv``.

Runs the policy (39 -> 10 contract, see ``bipedlab.rl.biped_velocity_env``) with frozen
commands and reports, per case, the body-frame forward speed after a 2 s settle,
falls, the per-step reward terms and the knee torque (mean, 95th percentile and max
of |actuator force|), and how often the 10 rad/s target slew limit of the runtime
supervisor cut a step (``slew_limited_fraction``: env-steps with at least one joint
limited, WP4-11) and the mean wanted target step per joint (rad). ``--dr`` adds
the same cases with domain randomization on (friction, mass, CoM, armature, IMU
tilt, encoder bias, random pushes). A zero-action run is the baseline.

The knee numbers re-check the WP4-01 requirement with a learned gait: the committed
model uses the XL330-class placeholder (0.52 N*m force limit), so a p95 close to
0.52 means the knee is saturating.

Usage: python scripts/eval_biped.py runs/biped/policy.onnx --dr --out reports/g4-3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab import bench
from bipedlab.rl import biped_velocity_env as bve

SETTLE_S = 2.0
TWISTS = [(0.15, 0.0, 0.0), (0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.8)]


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


def run_case(
    policy: BatchOnnxPolicy | None,
    twist: tuple[float, float, float],
    num_envs: int,
    seconds: float,
    seed: int,
    dr: bool,
) -> dict[str, Any]:
    env = bve.BipedVelocityEnv(
        num_envs=num_envs, seed=seed, episode_s=seconds + 1.0, common_step=2000 * 24, dr=dr
    )
    env.freeze_commands(np.array(twist))
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
        "wanted_target_step_mean": float(np.concatenate(wanted_step).mean()),
        "terms_per_step": {name: float(np.mean(v)) for name, v in terms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", type=Path, help="path to a 39->10 .onnx policy")
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dr", action="store_true", help="also run every case with DR on")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    policy = BatchOnnxPolicy(args.policy)
    cases = []
    for dr in [False, True] if args.dr else [False]:
        for twist in TWISTS:
            cases.append(run_case(policy, twist, args.envs, args.seconds, args.seed, dr))
        cases.append(run_case(None, TWISTS[0], args.envs, args.seconds, args.seed, dr))
    report = {"environment": bench.environment(), "policy": str(args.policy), "cases": cases}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
