"""Pinned upstream checkouts: lockfile parsing, fetching, and path lookup.

Upstream repositories are fetched at an exact commit into ``third_party/`` and
verified after checkout. Nothing from them is imported or executed; callers only
read data files (MJCF, ONNX) by path.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from bipedlab import REPO_ROOT

LOCKFILE = REPO_ROOT / "upstream.lock.toml"
THIRD_PARTY = REPO_ROOT / "third_party"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_URL_PREFIX = "https://github.com/"


class UpstreamError(RuntimeError):
    pass


@dataclass(frozen=True)
class UpstreamRepo:
    name: str
    url: str
    rev: str
    license: str

    @property
    def path(self) -> Path:
        return THIRD_PARTY / self.name


def load_lock(lockfile: Path = LOCKFILE) -> dict[str, UpstreamRepo]:
    data = tomllib.loads(lockfile.read_text())
    repos: dict[str, UpstreamRepo] = {}
    for name, entry in data.get("repos", {}).items():
        url, rev = entry["url"], entry["rev"]
        if not _SHA_RE.match(rev):
            raise UpstreamError(f"{name}: rev must be a full 40-char commit sha, got {rev!r}")
        if not url.startswith(_ALLOWED_URL_PREFIX) or not url.endswith(".git"):
            raise UpstreamError(f"{name}: url must be an https://github.com/... .git url")
        repos[name] = UpstreamRepo(name=name, url=url, rev=rev, license=entry["license"])
    return repos


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def checked_out_rev(repo: UpstreamRepo) -> str | None:
    if not (repo.path / ".git").exists():
        return None
    try:
        return _git("rev-parse", "HEAD", cwd=repo.path)
    except subprocess.CalledProcessError:
        return None


def fetch(repo: UpstreamRepo) -> Path:
    """Shallow-fetch ``repo`` at its pinned sha and verify HEAD matches."""
    if checked_out_rev(repo) == repo.rev:
        return repo.path
    repo.path.mkdir(parents=True, exist_ok=True)
    if not (repo.path / ".git").exists():
        _git("init", "-q", cwd=repo.path)
        _git("remote", "add", "origin", repo.url, cwd=repo.path)
    _git("fetch", "-q", "--depth", "1", "origin", repo.rev, cwd=repo.path)
    _git(
        "-c", "advice.detachedHead=false", "checkout", "-q", "--force", "FETCH_HEAD", cwd=repo.path
    )
    head = checked_out_rev(repo)
    if head != repo.rev:
        raise UpstreamError(f"{repo.name}: expected {repo.rev}, checked out {head}")
    return repo.path


def require(name: str) -> Path:
    """Path to a fetched upstream repo at its pinned sha, or raise with a fix hint."""
    repo = load_lock()[name]
    head = checked_out_rev(repo)
    if head != repo.rev:
        raise UpstreamError(
            f"{name} not available at pinned rev {repo.rev[:12]} (found {head}). "
            "Run: uv run python scripts/fetch_upstream.py"
        )
    return repo.path


MICRODUCK_MJCF_DIR = Path("src/mjlab_microduck/robot/microduck")


def microduck_scene(scene: str = "scene.xml") -> Path:
    """Path to an official Microduck MJCF scene inside the pinned microduck_rl."""
    return require("microduck_rl") / MICRODUCK_MJCF_DIR / scene
