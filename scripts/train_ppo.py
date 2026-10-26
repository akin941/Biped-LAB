"""WP3-03: short PPO training run on the WP3-02 velocity env (smoke test).

Trains rsl-rl-lib PPO (ADR-0012 settings) on ``MicroduckVelocityEnv`` for a fixed
number of iterations and records, per iteration, the mean per-step reward, the
fraction of envs that fell and the mean length of finished episodes. The smoke
criterion is that the mean reward of the last quarter of iterations is above that of
the first quarter. It checks that the env, the adapter and PPO learn together. It
does not judge walking quality.

Needs torch and rsl-rl-lib, which are not project dependencies yet:

    uv pip install torch==2.9.1 tensordict==0.10.0 rsl-rl-lib==5.0.1 \
        torchvision==0.24.1 onnxscript==0.5.7 gitpython==3.1.46

Usage: python scripts/train_ppo.py --envs 64 --iters 60 --seed 0 --out reports/ppo.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bipedlab import bench, upstream
from bipedlab.rl.parallel_env import ParallelBatchEnv
from bipedlab.rl.velocity_env import MicroduckVelocityEnv
from bipedlab.sim import microduck_policy as mp

HIDDEN = [512, 256, 128]
STEPS_PER_ENV = 24


class Recorder:
    """Per-iteration statistics, accumulated from env steps."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []
        self._reward: list[float] = []
        self._fell = 0
        self._lengths: list[int] = []
        self._terms: dict[str, list[float]] = {}

    def add(self, out: Any, episode_steps_before: np.ndarray) -> None:
        self._reward.append(float(out.reward.mean()))
        self._fell += int(out.terminated.sum())
        done = out.terminated | out.truncated
        self._lengths += [int(n) + 1 for n in episode_steps_before[done]]
        for k, v in out.terms.items():
            self._terms.setdefault(k, []).append(float(v.mean()))
        if len(self._reward) == STEPS_PER_ENV:
            self.rows.append(
                {
                    "reward_per_step": float(np.mean(self._reward)),
                    "falls": self._fell,
                    "episode_len": float(np.mean(self._lengths)) if self._lengths else np.nan,
                    **{k: float(np.mean(v)) for k, v in self._terms.items()},
                }
            )
            self._reward, self._fell, self._lengths, self._terms = [], 0, [], {}


def make_runner(env: Any, rec: Recorder) -> Any:
    from rsl_rl.env import VecEnv
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    class Adapter(VecEnv):
        def __init__(self) -> None:
            self.num_envs = env.num_envs
            self.num_actions = mp.ACT_DIM
            self.max_episode_length = env.max_episode_steps
            self.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
            self.device = "cpu"
            self.cfg: dict[str, Any] = {}

        def _obs(self, obs: np.ndarray, critic: np.ndarray) -> TensorDict:
            return TensorDict(
                {"policy": torch.from_numpy(obs), "critic": torch.from_numpy(critic)},
                batch_size=[env.num_envs],
            )

        def get_observations(self) -> TensorDict:
            return self._obs(env.obs, env.critic_obs)

        def step(
            self, actions: torch.Tensor
        ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
            before = env.episode_steps.copy()
            out = env.step(actions.numpy())
            rec.add(out, before)
            self.episode_length_buf = torch.from_numpy(env.episode_steps.copy())
            dones = torch.from_numpy(out.terminated | out.truncated).long()
            extras = {"time_outs": torch.from_numpy(out.truncated)}
            return self._obs(out.obs, out.critic_obs), torch.from_numpy(out.reward), dones, extras

    model = {
        "class_name": "MLPModel",
        "hidden_dims": HIDDEN,
        "activation": "elu",
        "obs_normalization": True,
    }
    cfg = {
        "num_steps_per_env": STEPS_PER_ENV,
        "save_interval": 10**9,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            **model,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": dict(model),
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }
    return OnPolicyRunner(Adapter(), cfg, log_dir=None, device="cpu")


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    scene = upstream.microduck_scene("scene_walk.xml")
    kwargs = {"seed": args.seed, "actuator": args.actuator}
    env: Any = (
        ParallelBatchEnv(
            scene, args.envs, args.workers, episode_s=20.0, env_cls=MicroduckVelocityEnv, **kwargs
        )  # fmt: skip
        if args.workers
        else MicroduckVelocityEnv(scene, args.envs, **kwargs)
    )
    rec = Recorder()
    start = time.perf_counter()
    try:
        make_runner(env, rec).learn(args.iters)
    finally:
        if isinstance(env, ParallelBatchEnv):
            env.close()
    wall = time.perf_counter() - start
    rewards = np.array([r["reward_per_step"] for r in rec.rows])
    q = max(1, len(rewards) // 4)
    first, last = float(rewards[:q].mean()), float(rewards[-q:].mean())
    return {
        "seed": args.seed,
        "envs": args.envs,
        "workers": args.workers,
        "actuator": args.actuator,
        "iterations": len(rec.rows),
        "samples": len(rec.rows) * STEPS_PER_ENV * args.envs,
        "wall_s": wall,
        "first_quarter_reward": first,
        "last_quarter_reward": last,
        "reward_rose": last > first,
        "history": rec.rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--workers", type=int, default=0, help="env processes (0 = serial)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actuator", choices=["mjcf", "bam"], default="mjcf")
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--require-rise", action="store_true", help="exit 1 unless the mean reward rose"
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    result = train(args)
    verdict = "rose" if result["reward_rose"] else "DID NOT RISE"
    report = {"environment": bench.environment(), "torch": torch.__version__, **result}
    print(
        f"seed {args.seed}: reward/step {result['first_quarter_reward']:.4f} -> "
        f"{result['last_quarter_reward']:.4f} ({verdict}), "
        f"{result['iterations']} it, {result['wall_s']:.0f} s"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.require_rise and not result["reward_rose"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
