"""Command line entry point: ``uv run bipedlab --help``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bipedlab import upstream
from bipedlab.sim import model as sim_model
from bipedlab.sim import rollout as sim_rollout

# Named shortcuts so the owner does not need to remember paths.
MODEL_ALIASES = {
    "toy": lambda: Path("simulation/custom-biped/toy_biped_v0.xml"),
    "microduck": lambda: upstream.microduck_scene("scene.xml"),
}


def _resolve(model: str) -> Path:
    return MODEL_ALIASES[model]() if model in MODEL_ALIASES else Path(model)


def _cmd_inspect(args: argparse.Namespace) -> int:
    path = _resolve(args.model)
    model, _ = sim_model.load(path)
    info = sim_model.describe(model, path)
    if args.json:
        print(json.dumps(info.to_dict(), indent=2))
        return 0
    print(
        f"{path}\n  timestep {info.timestep} s, mass {info.total_mass:.3f} kg, "
        f"nq {info.nq}, nv {info.nv}, actuators {info.nu}"
    )
    print("  actuators (kp, kv, forcerange):")
    for a in info.actuators:
        print(f"    {a.name:<20} kp={a.kp:<6g} kv={a.kv:<6g} force={a.forcerange}")
    print(f"  sensors: {', '.join(info.sensors)}")
    print(f"  keyframes: {', '.join(info.keyframes)}")
    return 0


def _cmd_stand(args: argparse.Namespace) -> int:
    path = _resolve(args.model)
    model, data = sim_model.load(path)
    result = sim_rollout.rollout(model, data, args.keyframe, args.seconds, args.hz)
    summary = {"model": str(path), "keyframe": args.keyframe, **result.summary()}
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_eval_policy(args: argparse.Namespace) -> int:
    from bipedlab import policies
    from bipedlab.sim import microduck_policy

    path = Path(args.policy)
    if not path.exists():
        path = policies.verified_path(args.policy)
    report = microduck_policy.evaluate(
        _resolve(args.model),
        path,
        init=args.init,
        post=microduck_policy.POSTPROCESS_PRESETS[args.post],
    )
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def _cmd_eval_behaviours(args: argparse.Namespace) -> int:
    from bipedlab import policies
    from bipedlab.sim import microduck_behaviours

    report = microduck_behaviours.evaluate(policies.verified_path, names=args.only or None)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from bipedlab import bench

    scene = _resolve(args.model)
    report: dict[str, object] = {
        "env_label": args.env_label,
        "environment": bench.environment(),
        "scene": str(scene),
        "physics_steps_per_s": round(bench.physics_steps_per_s(scene, seconds=args.seconds)),
    }
    if args.policy:
        from bipedlab import policies

        path = Path(args.policy)
        if not path.exists():
            path = policies.verified_path(args.policy)
        report["policy"] = path.name
        report["policy_loop_hz"] = round(bench.policy_loop_hz(scene, path, seconds=args.seconds))
    print(json.dumps(report, indent=2))
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    """Interactive viewer. On macOS run with `uv run mjpython -m bipedlab.cli view ...`."""
    import time

    import mujoco
    import mujoco.viewer

    scene = _resolve(args.model)
    if args.policy:
        from bipedlab import policies
        from bipedlab.sim import microduck_policy as mp

        path = Path(args.policy)
        if not path.exists():
            path = policies.verified_path(args.policy)
        model, data = mp.load_model_for_policy(scene)
        ctrl = mp.PolicyController(
            policy=mp.OnnxPolicy(path),
            observe=mp.ObservationBuilder(model),
            post=mp.POSTPROCESS_PRESETS[args.post],
        )
        ctrl.command[0:3] = args.twist
        decimation = mp.DECIMATION
    else:
        model, data = sim_model.load(scene)
        ctrl, decimation = None, 1
    mujoco.mj_resetDataKeyframe(model, data, model.key(args.keyframe).id)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            tick = time.perf_counter()
            if ctrl is not None:
                ctrl(model, data)
                ctrl.log.clear()
            mujoco.mj_step(model, data, nstep=decimation)
            viewer.sync()
            budget = decimation * model.opt.timestep - (time.perf_counter() - tick)
            if budget > 0:
                time.sleep(budget)
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    from bipedlab import policies
    from bipedlab.sim import microduck_policy as mp
    from bipedlab.telemetry import recorder as rec

    path = Path(args.policy)
    if not path.exists():
        path = policies.verified_path(args.policy)
    command = np.zeros(mp.CMD_DIM, dtype=np.float32)
    command[0:3] = args.twist
    holder: dict[str, rec.Recorder] = {}

    def wrap(model: mujoco.MjModel, control: rec.ControlFn) -> rec.ControlFn:
        holder["r"] = rec.Recorder(model, control, command=lambda: command)
        return holder["r"]

    mp.run(
        _resolve(args.model),
        mp.OnnxPolicy(path),
        args.seconds,
        twist=tuple(args.twist),
        init=args.init,
        recorder=wrap,
        post=mp.POSTPROCESS_PRESETS[args.post],
    )
    meta = {
        "policy": path.name,
        "twist": args.twist,
        "init": args.init,
        "postprocess": args.post,
        "actuator": "mjcf-position",
        "env_label": args.env_label,
    }
    out = holder["r"].save(args.out, meta=meta)
    if args.html:
        from bipedlab.telemetry import report

        report.write(out)
    summary = rec.summarize(holder["r"].arrays())
    print(json.dumps({"file": str(out), **meta, "summary": summary}, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from bipedlab.telemetry import report

    print(report.write(args.recording, args.out))
    return 0


POST_HELP = "robot-runtime action stage: none (upstream sim) or robotd-walking (0.9, LPF)"
POST_CHOICES = ["none", "robotd-walking"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bipedlab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="describe a MuJoCo model (joints, actuators, sensors)")
    p.add_argument("model", help="MJCF path or alias: " + ", ".join(MODEL_ALIASES))
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("stand", help="hold a keyframe pose open-loop and report stability")
    p.add_argument("model", help="MJCF path or alias: " + ", ".join(MODEL_ALIASES))
    p.add_argument("--keyframe", default="STAND")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--hz", type=float, default=50.0)
    p.set_defaults(func=_cmd_stand)

    p = sub.add_parser("eval-policy", help="run a Microduck ONNX policy over fixed scenarios")
    p.add_argument("policy", help="ONNX path, or a file name from the policy lock")
    p.add_argument("--model", default="microduck")
    p.add_argument("--out", help="also write the JSON report here")
    p.add_argument("--init", choices=["upstream", "settle"], default="upstream")
    p.add_argument("--post", choices=POST_CHOICES, default="none", help=POST_HELP)
    p.set_defaults(func=_cmd_eval_policy)

    from bipedlab.sim.microduck_behaviours import BEHAVIOURS

    p = sub.add_parser(
        "eval-behaviours", help="run sit/stand, ground pick, kicks and roulade policies"
    )
    p.add_argument("--only", nargs="+", choices=list(BEHAVIOURS), help="subset to run")
    p.add_argument("--out", help="also write the JSON report here")
    p.set_defaults(func=_cmd_eval_behaviours)

    p = sub.add_parser("record", help="run a Microduck policy and save per-step telemetry")
    p.add_argument("policy", help="ONNX path, or a file name from the policy lock")
    p.add_argument("--model", default="microduck")
    p.add_argument(
        "--twist", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ")
    )
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--init", choices=["upstream", "settle"], default="upstream")
    p.add_argument("--out", default="runs/record.npz")
    p.add_argument(
        "--env-label",
        default="unlabelled",
        choices=["unlabelled", "cloud", "gha-linux", "gha-macos", "mac-m5", "self-hosted"],
    )
    p.add_argument("--post", choices=POST_CHOICES, default="none", help=POST_HELP)
    p.add_argument("--html", action="store_true", help="also write an HTML report next to it")
    p.set_defaults(func=_cmd_record)

    p = sub.add_parser("report", help="turn a telemetry .npz into a self-contained HTML report")
    p.add_argument("recording")
    p.add_argument("--out")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("bench", help="measure CPU simulation speed (label the environment!)")
    p.add_argument("model", nargs="?", default="microduck")
    p.add_argument("--policy", help="also time the policy-in-the-loop control rate")
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument(
        "--env-label",
        required=True,
        choices=["cloud", "gha-linux", "gha-macos", "mac-m5", "self-hosted"],
    )
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser(
        "view", help="interactive MuJoCo viewer (needs a display; mjpython on macOS)"
    )
    p.add_argument("model", nargs="?", default="microduck")
    p.add_argument("--policy", help="ONNX path or locked policy name to run live")
    p.add_argument(
        "--twist", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("VX", "VY", "WZ")
    )
    p.add_argument("--post", choices=POST_CHOICES, default="none", help=POST_HELP)
    p.add_argument("--keyframe", default="STAND")
    p.set_defaults(func=_cmd_view)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
