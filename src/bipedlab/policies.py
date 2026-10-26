"""Pinned, checksum-verified pretrained policy files."""

from __future__ import annotations

import hashlib
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bipedlab import REPO_ROOT

LOCKFILE = REPO_ROOT / "learning/policies/microduck-policies.lock.toml"
CACHE = REPO_ROOT / "learning/policies/cache"
_HF = "https://huggingface.co"


class PolicyFileError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicyLock:
    repo: str
    revision: str
    files: dict[str, str]

    def url(self, name: str) -> str:
        return f"{_HF}/{self.repo}/resolve/{self.revision}/{name}"


def load_lock(lockfile: Path = LOCKFILE) -> PolicyLock:
    data = tomllib.loads(lockfile.read_text())
    lock = PolicyLock(repo=data["repo"], revision=data["revision"], files=dict(data["files"]))
    if len(lock.revision) != 40:
        raise PolicyFileError("policy revision must be a full 40-char commit sha")
    return lock


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_path(name: str, lock: PolicyLock | None = None, cache: Path = CACHE) -> Path:
    """Path to a cached policy whose sha256 matches the lock, else raise."""
    lock = lock or load_lock()
    if name not in lock.files:
        raise PolicyFileError(f"{name} is not in the policy lock")
    path = cache / name
    if not path.exists():
        raise PolicyFileError(
            f"{name} not downloaded; run: uv run python scripts/fetch_policies.py"
        )
    actual = sha256(path)
    if actual != lock.files[name]:
        raise PolicyFileError(f"{name}: sha256 {actual} != pinned {lock.files[name]}")
    return path


def fetch(name: str, lock: PolicyLock | None = None, cache: Path = CACHE) -> Path:
    lock = lock or load_lock()
    try:
        return verified_path(name, lock, cache)
    except PolicyFileError:
        pass
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f".{name}.part"
    with urllib.request.urlopen(lock.url(name), timeout=60) as resp:  # noqa: S310 (fixed https host)
        tmp.write_bytes(resp.read())
    actual = sha256(tmp)
    if actual != lock.files[name]:
        tmp.unlink()
        raise PolicyFileError(f"{name}: downloaded sha256 {actual} != pinned {lock.files[name]}")
    tmp.replace(cache / name)
    return cache / name
