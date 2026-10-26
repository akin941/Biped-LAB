"""G3.1: score an ONNX policy inside the WP3-02 velocity training env.

Runs the policy in ``MicroduckVelocityEnv`` with frozen commands and reports, per
case, the body-frame forward speed, falls, and the mean per-step value of every reward
term (already x weight x dt, so terms add up to the per-step reward). A zero-action
run is the baseline. Curriculum weights are the end-of-curriculum ones (2000 PPO
iterations), the regime a finished policy was trained in.

Pass criterion (ADR-0011, spec "Validation"): with the BAM actuator and a 0.4 m/s
command, ``alpha_walking`` reproduces upstream's sim number (~0.17 m/s, no falls) and
scores a reward profile consistent with a trained policy.

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
) -> dict[str, Any]:
    env = MicroduckVelocityEnv(
        scene, num_envs, seed=seed, actuator=actuator, episode_s=seconds + 1.0,
        noise=noise, common_step=2000 * 24,
    )  # fmt: skip
    env.freeze_commands(np.array(twist))
    steps = round(seconds * mp.CONTROL_HZ)
    settle = round(SETTLE_S * mp.CONTROL_HZ)
    alive = np.ones(num_envs, dtype=bool)
    vx: list[np.ndarray] = []
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
        for k, v in out.terms.items():
            terms.setdefault(k, []).append(float(v[alive].mean()) if alive.any() else np.nan)
        rewards.append(float(out.reward[alive].mean()) if alive.any() else np.nan)
    vx_arr = np.array(vx)
    return {
        "policy": "zero-action" if policy is None else "onnx",
        "actuator": actuator,
        "twist": list(twist),
        "noise": noise,
        "num_envs": num_envs,
        "seconds": seconds,
        "falls": int((~alive).sum()),
        "vx_mean": float(np.nanmean(vx_arr)) if np.isfinite(vx_arr).any() else None,
        "vx_per_env": [float(x) for x in np.nanmean(vx_arr, axis=0)]
        if np.isfinite(vx_arr).any()
        else [],
        "reward_per_step": float(np.nanmean(rewards)),
        "terms_per_step": {k: float(np.nanmean(v)) for k, v in terms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", help="policy file name in the verified cache")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--actuators", nargs="+", default=["bam", "mjcf"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    scene = upstream.microduck_scene("scene_walk.xml")
    policy = mp.OnnxPolicy(policies.verified_path(args.policy))
    cases = []
    for actuator in args.actuators:
        for twist in [(0.4, 0.0, 0.0), (0.0, 0.0, 0.0)]:
            for noise in (True, False):
                cases.append(run_case(scene, policy, actuator, twist, noise, args.envs,
                                      args.seconds, args.seed))  # fmt: skip
        cases.append(run_case(scene, None, actuator, (0.4, 0.0, 0.0), True, args.envs,
                              args.seconds, args.seed))  # fmt: skip
    report = {"environment": bench.environment(), "policy": args.policy, "cases": cases}
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
