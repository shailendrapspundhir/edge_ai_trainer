"""Shared pytest fixtures — isolate every test in its own temp dir + sqlite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every EAT_* path at tmp_path and use an in-memory sqlite db."""
    monkeypatch.setenv("EAT_ENV", "ci")
    monkeypatch.setenv("EAT_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("EAT_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EAT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("EAT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EAT_DB_URL", f"sqlite:///{tmp_path / 'eat-test.sqlite'}")
    # No real redis in tests; tests that need queueing should mock or skip.
    monkeypatch.setenv("EAT_REDIS_URL", "redis://127.0.0.1:6379/15")

    # Reset singletons that cached the previous settings.
    from eat import config
    config.get_settings.cache_clear()

    # Reset registry singletons (they cached repo_root()).
    from eat.registry import datasets, models, projects, recipes, targets
    for m in (datasets, models, recipes, targets):
        m.reload()
    # paths.repo_root is lru_cached
    from eat import paths
    paths.repo_root.cache_clear()
    return tmp_path
