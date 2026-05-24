"""Resolved filesystem layout for the project.

Everything that touches the disk goes through here. Paths are derived from the
working directory (the repo root) plus environment overrides — no hard-coding.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Best-effort guess at the repo root.

    Search upward from the current working directory for a marker file
    (pyproject.toml). Fall back to CWD if nothing matches — useful when running
    from inside a container that does not carry the marker.
    """
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cwd


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


def artifacts_dir() -> Path:
    return _env_path("EAT_ARTIFACTS_DIR", repo_root() / "artifacts")


def runs_dir() -> Path:
    return _env_path("EAT_RUNS_DIR", repo_root() / "runs")


def cache_dir() -> Path:
    return _env_path("EAT_CACHE_DIR", repo_root() / ".cache" / "eat")


def configs_dir() -> Path:
    return repo_root() / "configs"


def projects_dir() -> Path:
    return repo_root() / "projects"


def cloud_dir() -> Path:
    return repo_root() / "cloud"


def android_dir() -> Path:
    return repo_root() / "android"


def run_dir(run_id: str) -> Path:
    """Per-run scratch directory; created on demand."""
    p = runs_dir() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_artifacts_dir(run_id: str) -> Path:
    p = artifacts_dir() / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dirs() -> None:
    """Create the standard directories if missing. Safe to call repeatedly."""
    for fn in (artifacts_dir, runs_dir, cache_dir):
        fn().mkdir(parents=True, exist_ok=True)
