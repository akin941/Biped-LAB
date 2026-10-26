"""Regenerate ``simulation/custom-biped/biped_v1.xml`` from ``BipedParams`` (WP4-02).

Usage: uv run python scripts/gen_biped.py [--check]
``--check`` exits 1 if the committed file differs from the generator's output.
"""

from __future__ import annotations

import argparse

from bipedlab import REPO_ROOT
from bipedlab.design import biped

OUT = REPO_ROOT / "simulation/custom-biped/biped_v1.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()
    text = biped.to_mjcf()
    if args.check:
        if not OUT.is_file() or OUT.read_text() != text:
            raise SystemExit(f"{OUT} is stale; run scripts/gen_biped.py")
        print(f"{OUT} is up to date")
        return
    OUT.write_text(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
