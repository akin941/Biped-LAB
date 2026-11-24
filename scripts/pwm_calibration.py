"""Bench calibration of the DS3218MG pulse mapping (WP5-18, stage 3).

  plan   print the angles to command per joint and their nominal pulses (CSV template)
  fit    fit a filled CSV (joint,commanded_rad,measured_deg), print the report and the
         joints.rs snippet; exit 1 when a joint is refused or missing

The snippet is for review: applying it to the firmware needs the owner's approval
(ADR-0015). See bipedlab.pwm_calib for the model and limits.

Usage: uv run python scripts/pwm_calibration.py plan > readings.csv
       uv run python scripts/pwm_calibration.py fit readings.csv [--out report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from bipedlab import pwm_calib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--points", type=int, default=5)
    f = sub.add_parser("fit")
    f.add_argument("csv", type=Path)
    f.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cmd == "plan":
        w = csv.writer(sys.stdout)
        w.writerow(["joint", "commanded_rad", "measured_deg", "pulse_us"])
        for row in pwm_calib.plan(points=args.points):
            w.writerow([row["joint"], row["commanded_rad"], "", row["pulse_us"]])
        return
    report = pwm_calib.calibrate(pwm_calib.read_measurements(args.csv))
    text = json.dumps(report, indent=2)
    print(text)
    print()
    print(pwm_calib.rust_snippet(report))
    if args.out:
        args.out.write_text(text + "\n")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
