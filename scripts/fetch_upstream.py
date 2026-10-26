"""Fetch pinned upstream repositories into third_party/ (see upstream.lock.toml).

Usage: uv run python scripts/fetch_upstream.py [name ...]
"""

from __future__ import annotations

import sys

from bipedlab.upstream import fetch, load_lock


def main(argv: list[str]) -> int:
    repos = load_lock()
    names = argv or list(repos)
    unknown = [n for n in names if n not in repos]
    if unknown:
        print(f"unknown upstream(s): {', '.join(unknown)}; known: {', '.join(repos)}")
        return 2
    for name in names:
        repo = repos[name]
        path = fetch(repo)
        print(f"{name}: {repo.rev[:12]} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
