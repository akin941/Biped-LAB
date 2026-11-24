"""WP4-04: run the Rust runtime against the MuJoCo sim bridge and print the summary.

Usage:
  (cd runtime && cargo build --release)
  uv run python scripts/runtime_sim.py [--seconds 10] [--realtime] [--push-at 2 --push-dv 0 1.5 0]
      [--stall-at 1] [--out reports/runtime/stand.json] [--require-clean-exit]
      [--log runs/rt.bplog --env-label cloud] [-- RUNTIME_ARGS...]

A trained biped_v1 policy (WP4-10): ``-- --controller policy:policy.bpml`` after
``scripts/export_mlp.py policy.onnx --out policy.bpml``.

The DS3218MG build (G4.5): ``--model biped_ds3218 -- --controller policy:P.bpml
--joint-feedback commanded`` (PWM servos report nothing back, see WP4-39).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from bipedlab import runtime_bridge as rb
from bipedlab.rl import biped_velocity_env as bve

MODELS = {
    "biped_v1": bve.DEFAULT_SCENE,
    "biped_ds3218": bve.DS3218_SCENE,
    "biped_ds3218b": bve.DS3218B_SCENE,
    "biped_ds3218b_heavy": bve.DS3218B_HEAVY_SCENE,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, default=rb.DEFAULT_BINARY)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--realtime", action="store_true", help="runtime paces at 50 Hz wall clock")
    parser.add_argument("--push-at", type=float, default=None)
    parser.add_argument("--push-dv", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--stall-at", type=float, default=None)
    parser.add_argument(
        "--measure-from", type=float, default=2.0, help="s, start of the vx/distance window"
    )
    parser.add_argument("--torque-limit", type=float, default=None, help="N*m (WP4-30)")
    parser.add_argument(
        "--motor-no-load-speed", type=float, default=None,
        help="rad/s: DC-motor torque-speed curve on, as in training (WP4-30)",
    )  # fmt: skip
    parser.add_argument(
        "--init-seed", type=int, default=None,
        help="perturbed start (joint noise, heading, lift) as the env reset (WP4-36)",
    )  # fmt: skip
    parser.add_argument(
        "--model", choices=sorted(MODELS), default="biped_v1",
        help="sim model: biped_v1 or the DS3218MG build biped_ds3218 (G4.5)",
    )  # fmt: skip
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="runtime telemetry log (WP4-13); also writes LOG.npz/.json/.html via runtime_log",
    )
    parser.add_argument("--env-label", default="unlabelled", help="cloud / gha-linux / mac-m5")
    parser.add_argument(
        "--require-clean-exit",
        action="store_true",
        help="exit 1 unless the runtime exits 0 after serving every period",
    )
    parser.add_argument("runtime_args", nargs="*", help="extra biped-runtime arguments after --")
    args = parser.parse_args()
    cfg = rb.BridgeConfig(seconds=args.seconds, push_at=args.push_at,
                          push_dv=tuple(args.push_dv), stall_at=args.stall_at,
                          measure_from=args.measure_from, torque_limit=args.torque_limit,
                          motor_no_load_speed=args.motor_no_load_speed,
                          init_seed=args.init_seed)  # fmt: skip
    extra = [*args.runtime_args, *(["--realtime"] if args.realtime else [])]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        extra += ["--log", str(args.log)]
    model = mujoco.MjModel.from_xml_path(str(MODELS[args.model]))
    result = rb.run_with_runtime(args.binary, extra, cfg, model=model)
    result["model"] = args.model
    if args.log and args.log.exists():
        from bipedlab.design import biped
        from bipedlab.telemetry import report, runtime_log

        names = [f"{s}_{j}" for s in biped.SIDES for j in biped.LEG_JOINTS]
        npz = runtime_log.convert(args.log, args.log.with_suffix(".npz"), names,
                                  {"env_label": args.env_label})  # fmt: skip
        result["telemetry"] = {"npz": str(npz), "html": str(report.write(npz))}
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    if args.require_clean_exit:
        sim = result["sim"]
        periods = round(args.seconds / cfg.control_dt)
        if result["exit_code"] != 0 or "error" in sim or sim.get("steps") != periods:
            raise SystemExit(f"runtime did not run cleanly: exit {result['exit_code']}")


if __name__ == "__main__":
    main()
