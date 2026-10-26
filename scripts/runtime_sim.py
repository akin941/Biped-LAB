"""WP4-04: run the Rust runtime against the MuJoCo sim bridge and print the summary.

Usage:
  (cd runtime && cargo build --release)
  uv run python scripts/runtime_sim.py [--seconds 10] [--realtime] [--push-at 2 --push-dv 0 1.5 0]
      [--stall-at 1] [--out reports/runtime/stand.json] [-- RUNTIME_ARGS...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bipedlab import runtime_bridge as rb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, default=rb.DEFAULT_BINARY)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--realtime", action="store_true", help="runtime paces at 50 Hz wall clock")
    parser.add_argument("--push-at", type=float, default=None)
    parser.add_argument("--push-dv", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--stall-at", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("runtime_args", nargs="*", help="extra biped-runtime arguments after --")
    args = parser.parse_args()
    cfg = rb.BridgeConfig(seconds=args.seconds, push_at=args.push_at,
                          push_dv=tuple(args.push_dv), stall_at=args.stall_at)  # fmt: skip
    extra = [*args.runtime_args, *(["--realtime"] if args.realtime else [])]
    result = rb.run_with_runtime(args.binary, extra, cfg)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
