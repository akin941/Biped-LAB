"""WP5-03: write the runtime's joint contract (the bridge ``Hello``) to a file.

On hardware there is no sim to send the ``Hello``, so ``biped-runtime --device``
reads it from a file: control period, joint ranges, stand pose, servo gain and
torque limit, exactly as the sim bridge would send them for the same model.

Usage:
  uv run python scripts/export_hello.py --torque-limit 1.91 \
      --out hardware/calibration/biped_v1_sts3215.hello
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bipedlab import runtime_bridge as rb


def hello_bytes(torque_limit: float | None) -> bytes:
    return rb.SimBridge(cfg=rb.BridgeConfig(torque_limit=torque_limit)).hello()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--torque-limit", type=float, default=None, help="N*m, as in runtime_sim")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_bytes(hello_bytes(args.torque_limit))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
