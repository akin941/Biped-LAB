"""WP3-03: short PPO training run on the WP3-02 velocity env (smoke test).

Trains rsl-rl-lib PPO (ADR-0012 settings) on ``MicroduckVelocityEnv`` (``--robot
microduck``, default) or on our own biped's ``BipedVelocityEnv`` (``--robot biped_v1``,
G4.3 preparation) for a fixed number of iterations and records, per iteration, the mean
per-step reward, the fraction of envs that fell and the mean length of finished
episodes. The smoke criterion is that the mean reward of the last quarter of iterations
is above that of the first quarter. For biped_v1 each iteration also records the knee
torque (mean and max of |actuator force|), to re-check the WP4-01 knee requirement. It
checks that the env, the adapter and PPO learn together. It does not judge walking
quality.

Needs torch and rsl-rl-lib, which are not project dependencies yet:

    uv pip install torch==2.9.1 tensordict==0.10.0 rsl-rl-lib==5.0.1 \
        torchvision==0.24.1 onnxscript==0.5.7 gitpython==3.1.46

Usage: python scripts/train_ppo.py --envs 64 --iters 60 --seed 0 --out reports/ppo.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bipedlab import bench, upstream
from bipedlab.rl import biped_velocity_env as bve
from bipedlab.rl.parallel_env import ParallelBatchEnv
from bipedlab.rl.velocity_env import MicroduckVelocityEnv
from bipedlab.sim import microduck_policy as mp

HIDDEN = [512, 256, 128]
STEPS_PER_ENV = 24
# Initial action noise for biped_v1 (WP4-11). Its actions are scaled by 0.5 rad, so 0.5
# explores 0.25 rad per joint; 1.0 (Microduck's value at scale 1.0) made the
# action-rate penalty of pure noise outweigh standing and the policy learned to jitter.
BIPED_INIT_STD = 0.5
# --robot biped_v1 --model: the committed scenes (G4.5 adds the DS3218MG build).
BIPED_SCENES = {
    "biped_v1": bve.DEFAULT_SCENE,
    "biped_ds3218": bve.DS3218_SCENE,
    "biped_ds3218b": bve.DS3218B_SCENE,
    "biped_ds3218b_heavy": bve.DS3218B_HEAVY_SCENE,
}


class Recorder:
    """Per-iteration statistics, accumulated from env steps."""

    def __init__(self) -> None:
        self.rows: list[dict[str, float]] = []
        self._reward: list[float] = []
        self._fell = 0
        self._lengths: list[int] = []
        self._terms: dict[str, list[float]] = {}
        self._knee: list[np.ndarray] = []

    def add(self, out: Any, episode_steps_before: np.ndarray) -> None:
        self._reward.append(float(out.reward.mean()))
        self._fell += int(out.terminated.sum())
        done = out.terminated | out.truncated
        self._lengths += [int(n) + 1 for n in episode_steps_before[done]]
        for k, v in out.terms.items():
            self._terms.setdefault(k, []).append(float(v.mean()))
        torque = getattr(out, "joint_torque", None)
        if torque is not None:
            self._knee.append(torque[:, bve.KNEES])
        if len(self._reward) == STEPS_PER_ENV:
            row = {
                "reward_per_step": float(np.mean(self._reward)),
                "falls": self._fell,
                "episode_len": float(np.mean(self._lengths)) if self._lengths else np.nan,
                **{k: float(np.mean(v)) for k, v in self._terms.items()},
            }
            if self._knee:
                knee = np.stack(self._knee)
                row["knee_torque_mean"] = float(knee.mean())
                row["knee_torque_max"] = float(knee.max())
            self.rows.append(row)
            self._reward, self._fell, self._lengths, self._terms = [], 0, [], {}
            self._knee = []


def make_runner(
    env: Any,
    rec: Recorder,
    act_dim: int = mp.ACT_DIM,
    init_std: float = 1.0,
    symmetry: str | None = None,
    mirror_loss_coeff: float = 1.0,
    desired_kl: float = 0.01,
) -> Any:
    from rsl_rl.env import VecEnv
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    class Adapter(VecEnv):
        def __init__(self) -> None:
            self.num_envs = env.num_envs
            self.num_actions = act_dim
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
                "init_std": init_std,
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
            "desired_kl": desired_kl,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": symmetry_cfg(symmetry, mirror_loss_coeff, act_dim),
        },
    }
    runner = OnPolicyRunner(Adapter(), cfg, log_dir=None, device="cpu")
    # Keep PPO's losses (surrogate, value, entropy, symmetry) in the history row of the
    # iteration they belong to: the row is written during the rollout, the update follows.
    update = runner.alg.update

    def update_and_record(*a: Any, **kw: Any) -> Any:
        losses = update(*a, **kw)
        if rec.rows and isinstance(losses, dict):
            rec.rows[-1].update({f"loss_{k}": float(v) for k, v in losses.items() if v is not None})
        return losses

    runner.alg.update = update_and_record
    return runner


def symmetry_cfg(
    mode: str | None, mirror_loss_coeff: float = 1.0, act_dim: int = bve.ACT_DIM
) -> dict[str, Any] | None:
    """Left/right mirror (WP4-23 biped_v1, WP3-16 Microduck): "aug" data augmentation,
    "loss" mirror loss.

    Augmentation gives each mirrored sample the original's old log-probability, so on an
    asymmetric warm start (MAC-14) the PPO ratios of the mirrored half are far from 1:
    MAC-23 (mac-m5) degraded from iteration ~100. Fine-tune with "loss" instead (WP4-24).
    """
    if mode is None:
        return None
    from bipedlab.rl import biped_symmetry, microduck_symmetry

    sym = biped_symmetry if act_dim == bve.ACT_DIM else microduck_symmetry
    return {
        "use_data_augmentation": mode in ("aug", "both"),
        "use_mirror_loss": mode in ("loss", "both"),
        "mirror_loss_coeff": mirror_loss_coeff,
        "data_augmentation_func": sym.data_augmentation,
    }


def save_checkpoint(
    runner: Any, args: argparse.Namespace, iterations: int, common_step: int, rec: Recorder
) -> None:
    """Write ``model_<iterations>.pt`` (rsl-rl format plus our counters) and the history."""
    args.run_dir.mkdir(parents=True, exist_ok=True)
    infos = {"iterations": iterations, "common_step": common_step}
    runner.save(str(args.run_dir / f"model_{iterations}.pt"), infos=infos)
    # One history file per run segment, so a resumed run keeps the earlier one.
    start = iterations - len(rec.rows)
    (args.run_dir / f"history_from_{start}.json").write_text(json.dumps(rec.rows) + "\n")


def export_onnx(runner: Any, path: Path) -> None:
    """Actor (with its observation normaliser) as ONNX (61 -> 14, or 39 -> 10 for biped_v1).

    Uses the TorchScript exporter at opset 17, as upstream microduck_rl does
    (``dynamo=False``): torch 2.9's default dynamo exporter fails to convert rsl-rl's
    actor to an older opset.
    """
    model = runner.alg.get_policy().as_onnx(verbose=False)
    model.to("cpu")
    model.eval()
    torch.onnx.export(
        model,
        model.get_dummy_inputs(),
        str(path),
        export_params=True,
        opset_version=17,
        input_names=model.input_names,
        output_names=model.output_names,
        dynamo=False,
    )


def write_manifest(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """``run.json``: config, seed, git revision, metrics and artifact sha256 (G3.x)."""
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    artifacts = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(args.run_dir.iterdir())
        if f.suffix in (".onnx", ".pt")
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {
        "git_revision": rev,
        "git_dirty": dirty,
        "config": config,
        "metrics": {k: v for k, v in report.items() if k != "history"},
        "artifacts_sha256": artifacts,
    }
    (args.run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    biped_v1 = args.robot == "biped_v1"
    if biped_v1 and args.actuator != "mjcf":
        raise SystemExit("biped_v1 has MJCF position actuators only")
    if biped_v1 and args.track_lin_std2 is not None:
        raise SystemExit("--track-lin-std2 is a Microduck option (biped_v1 uses 0.05)")
    if args.desired_kl <= 0:
        raise SystemExit("--desired-kl must be > 0")
    if biped_v1 and args.turn_fraction is not None:
        raise SystemExit("--turn-fraction is a Microduck option")
    if args.mirror_loss_coeff <= 0:
        raise SystemExit("--mirror-loss-coeff must be > 0")
    if biped_v1 and args.posture_gate:
        raise SystemExit("--posture-gate is a Microduck option")
    if biped_v1 and args.track_lin_ref_speed is not None:
        raise SystemExit("--track-lin-ref-speed is a Microduck option")
    if biped_v1 and args.track_ang_ref_speed is not None:
        raise SystemExit("--track-ang-ref-speed is a Microduck option")
    if biped_v1 and args.forward_min_vx is not None:
        raise SystemExit("--forward-min-vx is a Microduck option")
    if not biped_v1 and (args.model != "biped_v1" or args.joint_feedback != "measured"):
        raise SystemExit("--model / --joint-feedback are biped_v1 options")
    if not biped_v1 and args.servo_dr:
        raise SystemExit("--servo-dr is a biped_v1 option")
    if not biped_v1 and args.track_yaw_std2 is not None:
        raise SystemExit("--track-yaw-std2 is a biped_v1 option")
    if not biped_v1 and (args.torque_limit is not None or args.motor_no_load_speed is not None):
        raise SystemExit("--torque-limit / --motor-no-load-speed are biped_v1 options")
    if args.rough_floor < 0 or args.rough_floor_ramp_it < 0:
        raise SystemExit("--rough-floor and --rough-floor-ramp-it must be >= 0")
    if not 0.0 <= args.rough_floor_flat_fraction <= 1.0:
        raise SystemExit("--rough-floor-flat-fraction must be in [0, 1]")
    if args.rough_floor and not biped_v1:
        raise SystemExit("--rough-floor is a biped_v1 option")
    if args.motor_no_load_speed is not None and args.motor_no_load_speed <= 0:
        raise SystemExit("--motor-no-load-speed must be > 0")
    if args.torque_limit is not None and args.torque_limit <= 0:
        raise SystemExit("--torque-limit must be > 0")
    resume: dict[str, Any] | None = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    # Curricula continue from where the checkpoint left them.
    infos = (resume or {}).get("infos") or {}
    common_step = int(infos.get("common_step", 0))
    first_iter = int(infos.get("iterations", 0))
    kwargs: dict[str, Any] = {"seed": args.seed, "common_step": common_step, "dr": args.dr}
    env_cls: type
    if biped_v1:
        scene, env_cls, act_dim = BIPED_SCENES[args.model], bve.BipedVelocityEnv, bve.ACT_DIM
        kwargs["joint_feedback"] = args.joint_feedback
        kwargs["servo_dr"] = args.servo_dr
        if args.dr_delay_max is not None:
            kwargs["dr_delay_max"] = args.dr_delay_max
        if args.dr_friction is not None:
            kwargs["dr_foot_friction"] = tuple(args.dr_friction)
        if args.track_yaw_std2 is not None:
            kwargs["track_yaw_std2"] = args.track_yaw_std2
        if args.torque_limit is not None:
            kwargs["torque_limit"] = args.torque_limit
        if args.motor_no_load_speed is not None:
            kwargs["motor_no_load_speed"] = args.motor_no_load_speed
        if args.rough_floor:
            kwargs["rough_floor"] = args.rough_floor
            kwargs["rough_floor_ramp"] = args.rough_floor_ramp_it * STEPS_PER_ENV
            # A field the evaluation (seed 0 by default) never shows.
            kwargs["rough_floor_seed"] = args.seed + 1000
            kwargs["rough_floor_flat_fraction"] = args.rough_floor_flat_fraction
    elif args.dr_delay_max is not None or args.dr_friction is not None:
        raise SystemExit("--dr-delay-max / --dr-friction are biped_v1 options")
    else:
        if args.track_lin_std2 is not None:
            kwargs["track_lin_std2"] = args.track_lin_std2
        if args.forward_min_vx is not None:
            kwargs["forward_min_vx"] = args.forward_min_vx
        if args.posture_gate:
            kwargs["posture_gate"] = True
        if args.track_lin_ref_speed is not None:
            kwargs["track_lin_ref_speed"] = args.track_lin_ref_speed
        if args.track_ang_ref_speed is not None:
            kwargs["track_ang_ref_speed"] = args.track_ang_ref_speed
        if args.turn_fraction is not None:
            kwargs["turn_fraction"] = args.turn_fraction
        scene = upstream.microduck_scene("scene_walk.xml")
        env_cls, act_dim = MicroduckVelocityEnv, mp.ACT_DIM
        kwargs["actuator"] = args.actuator
    env: Any = (
        ParallelBatchEnv(scene, args.envs, args.workers, episode_s=20.0, env_cls=env_cls, **kwargs)
        if args.workers
        else env_cls(scene=scene, num_envs=args.envs, **kwargs)
    )
    rec = Recorder()
    start = time.perf_counter()
    try:
        if args.init_std is None:
            args.init_std = BIPED_INIT_STD if biped_v1 else 1.0
        runner = make_runner(
            env, rec, act_dim, args.init_std, args.symmetry, args.mirror_loss_coeff,
            args.desired_kl,
        )  # fmt: skip
        if args.resume:
            runner.load(str(args.resume))
        done = 0
        while done < args.iters:
            chunk = min(args.save_every or args.iters, args.iters - done)
            runner.learn(chunk)
            done += chunk
            if args.run_dir:
                total = first_iter + done
                save_checkpoint(runner, args, total, common_step + done * STEPS_PER_ENV, rec)
        if args.run_dir:
            args.run_dir.mkdir(parents=True, exist_ok=True)  # also for --iters 0 exports
            export_onnx(runner, args.run_dir / "policy.onnx")
    finally:
        if isinstance(env, ParallelBatchEnv):
            env.close()
    wall = time.perf_counter() - start
    rewards = np.array([r["reward_per_step"] for r in rec.rows])
    q = max(1, len(rewards) // 4)
    first, last = float(rewards[:q].mean()), float(rewards[-q:].mean())
    knee = [r["knee_torque_max"] for r in rec.rows if "knee_torque_max" in r]
    return {
        "robot": args.robot,
        "seed": args.seed,
        "envs": args.envs,
        "workers": args.workers,
        "actuator": args.actuator,
        "dr": args.dr,
        "servo_dr": args.servo_dr,
        "init_std": args.init_std,
        "dr_delay_max": args.dr_delay_max,
        "dr_friction": args.dr_friction,
        "track_lin_std2": args.track_lin_std2,
        "track_yaw_std2": args.track_yaw_std2,
        "torque_limit": args.torque_limit,
        "motor_no_load_speed": args.motor_no_load_speed,
        "rough_floor": args.rough_floor,
        "rough_floor_ramp_it": args.rough_floor_ramp_it,
        "rough_floor_flat_fraction": args.rough_floor_flat_fraction,
        "forward_min_vx": args.forward_min_vx,
        "posture_gate": args.posture_gate,
        "track_lin_ref_speed": args.track_lin_ref_speed,
        "track_ang_ref_speed": args.track_ang_ref_speed,
        "turn_fraction": args.turn_fraction,
        "desired_kl": args.desired_kl,
        "symmetry": args.symmetry,
        "mirror_loss_coeff": args.mirror_loss_coeff if args.symmetry else None,
        "iterations": len(rec.rows),
        "samples": len(rec.rows) * STEPS_PER_ENV * args.envs,
        "wall_s": wall,
        "first_quarter_reward": first,
        "last_quarter_reward": last,
        "reward_rose": last > first,
        "knee_torque_max": max(knee) if knee else None,
        "history": rec.rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", choices=["microduck", "biped_v1"], default="microduck")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--workers", type=int, default=0, help="env processes (0 = serial)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actuator", choices=["mjcf", "bam"], default="mjcf")
    parser.add_argument("--dr", action="store_true", help="domain randomization on")
    parser.add_argument(
        "--dr-delay-max", type=int, default=None,
        help="biped_v1 DR action delay 0..N periods (default 2; 0 = off, WP4-18 ablation)",
    )  # fmt: skip
    parser.add_argument(
        "--dr-friction", type=float, nargs=2, default=None, metavar=("LO", "HI"),
        help="biped_v1 DR foot friction range (default 0.7 1.3)",
    )  # fmt: skip
    parser.add_argument(
        "--track-lin-std2", type=float, default=None,
        help="Microduck linear tracking kernel std^2 (default 0.1, upstream; WP3-12)",
    )  # fmt: skip
    parser.add_argument(
        "--symmetry", choices=("aug", "loss", "both"), default=None,
        help="left/right mirror: data augmentation, mirror loss or both (WP4-23, WP3-16)",
    )  # fmt: skip
    parser.add_argument(
        "--mirror-loss-coeff", type=float, default=1.0,
        help="weight of the mirror loss with --symmetry loss/both (rsl-rl default 1.0)",
    )  # fmt: skip
    parser.add_argument(
        "--posture-gate", action="store_true",
        help="Microduck: scale posture rewards by tracking under a nonzero command (WP3-14)",
    )  # fmt: skip
    parser.add_argument(
        "--track-lin-ref-speed", type=float, default=None,
        help="Microduck: shrink the linear kernel width below this command speed (WP3-15)",
    )  # fmt: skip
    parser.add_argument(
        "--turn-fraction", type=float, default=None,
        help="Microduck share of turn-only command draws (default 0.15, upstream; WP3-19)",
    )  # fmt: skip
    parser.add_argument(
        "--desired-kl", type=float, default=0.01,
        help="PPO adaptive-lr KL target (default 0.01); lower is more conservative (WP3-19)",
    )  # fmt: skip
    parser.add_argument(
        "--track-ang-ref-speed", type=float, default=None,
        help="Microduck: shrink the yaw kernel width below this yaw-rate command (WP3-17)",
    )  # fmt: skip
    parser.add_argument(
        "--forward-min-vx", type=float, default=None,
        help="Microduck floor of forward-only command draws (default 0.3, upstream; WP3-13)",
    )  # fmt: skip
    parser.add_argument(
        "--track-yaw-std2", type=float, default=None,
        help="biped_v1 yaw-rate tracking kernel std^2 (default 0.5; WP4-22)",
    )  # fmt: skip
    parser.add_argument(
        "--torque-limit", type=float, default=None,
        help="biped_v1 actuator force limit in N*m (default: the model's 0.52; WP4-26)",
    )  # fmt: skip
    parser.add_argument(
        "--motor-no-load-speed", type=float, default=None,
        help="biped_v1 DC-motor no-load speed in rad/s: torque-speed curve on (WP4-28)",
    )  # fmt: skip
    parser.add_argument(
        "--model", choices=sorted(BIPED_SCENES), default="biped_v1",
        help="with --robot biped_v1: the model, biped_v1 or the DS3218MG build (G4.5)",
    )  # fmt: skip
    parser.add_argument(
        "--joint-feedback", choices=bve.JOINT_FEEDBACK, default="measured",
        help="biped_v1 actor joint terms: measured, or commanded targets for PWM servos (G4.5)",
    )  # fmt: skip
    parser.add_argument(
        "--servo-dr", action="store_true",
        help="biped_v1: PWM servo offset / deadband / stiffness draws per reset (WP4-41)",
    )  # fmt: skip
    parser.add_argument(
        "--rough-floor", type=float, default=0.0,
        help="biped_v1 uneven tile floor, peak-to-peak height in m (default flat; WP4-32)",
    )  # fmt: skip
    parser.add_argument(
        "--rough-floor-ramp-it", type=int, default=0,
        help="iterations over which --rough-floor ramps up from flat (default 0 = at once)",
    )  # fmt: skip
    parser.add_argument(
        "--rough-floor-flat-fraction", type=float, default=0.0,
        help="share of --rough-floor episodes run on a flat floor (WP4-33, default 0)",
    )  # fmt: skip
    parser.add_argument(
        "--init-std",
        type=float,
        default=None,
        help="initial action noise std (default: 1.0, biped_v1: BIPED_INIT_STD)",
    )
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--run-dir", type=Path, default=None, help="save checkpoints and policy.onnx here"
    )
    parser.add_argument("--save-every", type=int, default=0, help="iterations per checkpoint")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
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
        f"{args.robot} seed {args.seed}: reward/step {result['first_quarter_reward']:.4f} -> "
        f"{result['last_quarter_reward']:.4f} ({verdict}), "
        f"{result['iterations']} it, {result['wall_s']:.0f} s"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.run_dir:
        write_manifest(args, report)
    if args.require_rise and not result["reward_rose"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
