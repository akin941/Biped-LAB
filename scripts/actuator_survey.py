"""WP4-03: fit of candidate servos to the WP4-01 joint requirements.

Prints, for every joint type, which candidates pass the 2x torque and speed margins
and the loaded-speed check (see ``bipedlab.design.actuators``). ``--out`` stores
the full table as JSON. Prices are a dated retail snapshot; nothing is bought.

Usage:
  uv run python scripts/actuator_survey.py [--out reports/actuators.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bipedlab.design import actuators as act


def table() -> dict[str, Any]:
    return {
        r.joint: {
            "requirement": asdict(r),
            "candidates": {f"{c.name} @ {c.volts:g} V": act.fit(c, r) for c in act.CANDIDATES},
        }
        for r in act.WP4_01
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = {"margin": act.MARGIN, "candidates": [asdict(c) for c in act.CANDIDATES],
              "joints": table()}  # fmt: skip
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    for joint, row in report["joints"].items():
        r = row["requirement"]
        print(f"\n{joint}: peak {r['peak_torque']:.3f} N*m, {r['peak_speed']:.2f} rad/s")
        for name, f in row["candidates"].items():
            mark = "PASS" if f["passes"] else "fail"
            print(f"  {mark} {name:24s} torque x{f['torque_margin']:.2f}"
                  f"  speed x{f['speed_margin']:.2f}"
                  f"  loaded {f['loaded_speed_rad_s']:5.2f} rad/s"
                  f"  rms/stall {f['rms_over_stall']:.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
