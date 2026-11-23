"""WP5-05: replay a hardware run's commands in the sim and compare the two logs.

Usage (after a Phase 5 stage 4-6 run with ``biped-runtime --device ... --log REAL``):
  (cd runtime && cargo build --release)
  uv run python scripts/sim_vs_real.py REAL.bplog --out reports/sim_vs_real.json \
      --torque-limit 1.91 --motor-no-load-speed 5.45 --env-label hw \
      -- --controller policy:policy.bpml --torque-fraction 0.3

Everything after ``--`` goes to the sim runtime; pass the same controller and
torque fraction the robot used. ``--sim-log`` keeps the sim run's log. Body
speed is not on the robot's log, so it is not compared here.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from bipedlab import runtime_bridge as rb
from bipedlab.telemetry import runtime_log, sim_vs_real


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("real_log", type=Path)
    parser.add_argument("--binary", type=Path, default=rb.DEFAULT_BINARY)
    parser.add_argument("--torque-limit", type=float, default=None, help="N*m (WP4-30)")
    parser.add_argument("--motor-no-load-speed", type=float, default=None, help="rad/s (WP4-30)")
    parser.add_argument("--tol", type=float, default=0.01, help="command segment tolerance")
    parser.add_argument("--sim-log", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env-label", default="unlabelled", help="where the real log came from")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:cut])
    args.runtime_args = argv[cut + 1 :]

    real_arrays, real_header = runtime_log.read(args.real_log)
    segs = sim_vs_real.command_segments(real_arrays["time"], real_arrays["command"], args.tol)
    seconds = float(real_arrays["time"][-1] - real_arrays["time"][0]) + real_header["control_dt"]
    with tempfile.TemporaryDirectory() as tmp:
        sim_log = args.sim_log or Path(tmp) / "sim.bplog"
        cfg = rb.BridgeConfig(seconds=seconds, torque_limit=args.torque_limit,
                              motor_no_load_speed=args.motor_no_load_speed)  # fmt: skip
        extra = [*args.runtime_args, *sim_vs_real.runtime_command_args(segs), "--log", str(sim_log)]
        run = rb.run_with_runtime(args.binary, extra, cfg)
        if not sim_log.exists():
            raise SystemExit(f"sim run wrote no log: {run.get('stderr')}")
        sim_arrays, sim_header = runtime_log.read(sim_log)
    result = {
        "real_env": args.env_label,
        "sim_env": "sim bridge on this machine",
        "command_segments": len(segs),
        **sim_vs_real.compare(
            sim_vs_real.metrics(real_arrays, real_header),
            sim_vs_real.metrics(sim_arrays, sim_header),
        ),  # fmt: skip
        "sim_runtime": run["runtime"],
    }
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
