"""A/B env steps/s of MicroduckBatchEnv against another revision's batch_env.py (WP3-07).

Usage: python scripts/ab_batch_env.py OLD_BATCH_ENV.py [--num-envs 128] [--rounds 3]
Alternates old and new so both see the same machine load; prints rates per round.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from bipedlab import bench, upstream
from bipedlab.rl.batch_env import MicroduckBatchEnv


def rate(cls: type, scene: Path, num_envs: int, seconds: float) -> float:
    env = cls(scene, num_envs)
    actions = np.random.default_rng(0).uniform(-0.2, 0.2, (32, num_envs, 14)).astype(np.float32)
    env.step(actions[0])
    n, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        env.step(actions[n % 32])
        n += 1
    return n * num_envs / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("old_batch_env", args.old)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old  # dataclasses look the module up while decorating
    spec.loader.exec_module(old)
    scene = upstream.microduck_scene("scene.xml")
    rounds = []
    for _ in range(args.rounds):
        a = rate(old.MicroduckBatchEnv, scene, args.num_envs, args.seconds)
        b = rate(MicroduckBatchEnv, scene, args.num_envs, args.seconds)
        rounds.append({"old": round(a), "new": round(b), "ratio": round(b / a, 3)})
    ratios = [r["ratio"] for r in rounds]
    report = {"environment": bench.environment(), "num_envs": args.num_envs,
              "rounds": rounds, "median_ratio": float(np.median(ratios))}  # fmt: skip
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
