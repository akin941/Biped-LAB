"""Regenerate the committed biped models from ``BipedParams`` (WP4-02, G4.5).

Writes ``simulation/custom-biped/<name>.xml`` for every entry of
``bipedlab.design.biped.VARIANTS`` (biped_v1 and the DS3218MG build, biped_ds3218).

Usage: uv run python scripts/gen_biped.py [--check]
``--check`` exits 1 if the committed file differs from the generator's output.
"""

from __future__ import annotations

import argparse

from bipedlab import REPO_ROOT
from bipedlab.design import biped

OUT_DIR = REPO_ROOT / "simulation/custom-biped"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()
    for name, params in biped.VARIANTS.items():
        out = OUT_DIR / f"{name}.xml"
        text = biped.to_mjcf(params)
        if args.check:
            if not out.is_file() or out.read_text() != text:
                raise SystemExit(f"{out} is stale; run scripts/gen_biped.py")
            print(f"{out} is up to date")
            continue
        out.write_text(text)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
