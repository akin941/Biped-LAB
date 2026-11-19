"""G3.1: score an ONNX policy inside the WP3-02 velocity training env.

Runs the policy in ``MicroduckVelocityEnv`` with frozen commands and reports, per
case, the body-frame forward speed, falls, and the mean per-step value of every reward
term (already x weight x dt, so terms add up to the per-step reward). A zero-action
run is the baseline. Curriculum weights are the end-of-curriculum ones (2000 PPO
iterations), the regime a finished policy was trained in.

Pass criterion (ADR-0011, spec "Validation"): with the BAM actuator and a 0.4 m/s
command, ``alpha_walking`` reproduces upstream's sim number (~0.17 m/s, no falls) and
scores a reward profile consistent with a trained policy. ``--dr`` repeats the cases
with domain randomization on, the conditions ``alpha_walking`` was trained under.

Usage: python scripts/eval_velocity_env.py alpha_walking.onnx --out reports/g3-1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bipedlab import bench, policies, upstream
from bipedlab.rl.velocity_env import MicroduckVelocityEnv
from bipedlab.sim import microduck_policy as mp

SETTLE_S = 2.0


def run_case(
    scene: Path,
    policy: mp.OnnxPolicy | None,
    actuator: str,
    twist: tuple[float, float, float],
    noise: bool,
    num_envs: int,
    seconds: float,
    seed: int,
    dr: bool = False,
    reward_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = MicroduckVelocityEnv(
        scene, num_envs, seed=seed, actuator=actuator, episode_s=seconds + 1.0,
        noise=noise, common_step=2000 * 24, dr=dr, **(reward_kw or {}),
    )  # fmt: skip
    env.freeze_commands(np.array(twist))
    steps = round(seconds * mp.CONTROL_HZ)
    settle = round(SETTLE_S * mp.CONTROL_HZ)
    alive = np.ones(num_envs, dtype=bool)
    vx: list[np.ndarray] = []
    wz: list[np.ndarray] = []
    terms: dict[str, list[float]] = {}
    rewards: list[float] = []
    obs = env.obs
    for t in range(steps):
        if policy is None:
            actions = np.zeros((num_envs, mp.ACT_DIM), dtype=np.float32)
        else:
            actions = np.stack([policy(o) for o in obs])
        out = env.step(actions)
        alive &= ~out.terminated
        obs = out.obs
        if t >= settle:
            vx.append(np.where(alive, out.critic_obs[:, 61], np.nan))
            yaw = np.array([d.qvel[5] for d in env.datas])  # free joint, body frame
            wz.append(np.where(alive, yaw, np.nan))
        for k, v in out.terms.items():
            terms.setdefault(k, []).append(float(v[alive].mean()) if alive.any() else np.nan)
        rewards.append(float(out.reward[alive].mean()) if alive.any() else np.nan)
    vx_arr = np.array(vx)
    return {
        "policy": "zero-action" if policy is None else "onnx",
        "actuator": actuator,
        "twist": list(twist),
        "noise": noise,
        "dr": dr,
        "num_envs": num_envs,
        "seconds": seconds,
        "falls": int((~alive).sum()),
        "vx_mean": float(np.nanmean(vx_arr)) if np.isfinite(vx_arr).any() else None,
        "vx_per_env": [float(x) for x in np.nanmean(vx_arr, axis=0)]
        if np.isfinite(vx_arr).any()
        else [],
        "wz_mean": float(np.nanmean(wz)) if np.isfinite(np.array(wz)).any() else None,
        "reward_per_step": float(np.nanmean(rewards)),
        "terms_per_step": {k: float(np.nanmean(v)) for k, v in terms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "policy", help="policy file name in the verified cache, or a path to an .onnx file"
    )
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--actuators", nargs="+", default=["bam", "mjcf"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dr", action="store_true", help="domain randomization on (end-of-curriculum ranges)"
    )
    parser.add_argument(
        "--twist", type=float, nargs=3, action="append", metavar=("VX", "VY", "WZ"),
        help="command(s) to score instead of 0.4 forward and zero; repeatable (WP3-18)",
    )  # fmt: skip
    # WP3-18: score with the reward the policy was trained on, not the upstream one.
    parser.add_argument("--track-lin-std2", type=float, default=None)
    parser.add_argument("--posture-gate", action="store_true")
    parser.add_argument("--track-lin-ref-speed", type=float, default=None)
    parser.add_argument("--track-ang-ref-speed", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    reward_kw: dict[str, Any] = {"posture_gate": args.posture_gate}
    for key in ("track_lin_std2", "track_lin_ref_speed", "track_ang_ref_speed"):
        if getattr(args, key) is not None:
            reward_kw[key] = getattr(args, key)
    twists = [tuple(t) for t in args.twist] if args.twist else [(0.4, 0.0, 0.0), (0.0, 0.0, 0.0)]
    scene = upstream.microduck_scene("scene_walk.xml")
    local = Path(args.policy)
    path = local if local.suffix == ".onnx" and local.is_file() else None
    policy = mp.OnnxPolicy(path or policies.verified_path(args.policy))
    cases = []
    for actuator in args.actuators:
        for twist in twists:
            for noise in (True, False):
                cases.append(run_case(scene, policy, actuator, twist, noise, args.envs,
                                      args.seconds, args.seed, args.dr, reward_kw))  # fmt: skip
        cases.append(run_case(scene, None, actuator, twists[0], True, args.envs,
                              args.seconds, args.seed, args.dr, reward_kw))  # fmt: skip
    report = {
        "environment": bench.environment(), "policy": args.policy, "reward": reward_kw,
        "cases": cases,
    }  # fmt: skip
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
