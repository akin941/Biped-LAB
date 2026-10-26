"""Download pinned pretrained policies and verify their sha256.

Usage: uv run python scripts/fetch_policies.py [file.onnx ...]   (default: all)
"""

from __future__ import annotations

import sys

from bipedlab.policies import fetch, load_lock


def main(argv: list[str]) -> int:
    lock = load_lock()
    for name in argv or list(lock.files):
        path = fetch(name, lock)
        print(f"ok {name} sha256={lock.files[name][:12]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
