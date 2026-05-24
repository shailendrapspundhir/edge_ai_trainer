"""Environment sanity check: paths, redis, hf token, GPU, adb."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from eat.config import get_settings
from eat.paths import (
    artifacts_dir,
    cache_dir,
    configs_dir,
    projects_dir,
    repo_root,
    runs_dir,
)


def _check_redis(url: str) -> tuple[bool, str]:
    try:
        import redis
        r = redis.from_url(url, socket_connect_timeout=2)
        r.ping()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"unreachable ({exc})"


def _check_gpu() -> tuple[bool, str]:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return False, "nvidia-smi not found"
    try:
        out = subprocess.run(
            [nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return True, out or "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _check_torch_cuda() -> tuple[bool, str]:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True, f"cuda={torch.version.cuda}, devices={torch.cuda.device_count()}"
        return False, "torch installed, but cuda not available"
    except ImportError:
        return False, "torch not installed (run `make install-train`)"


def _check_adb(adb_bin: str) -> tuple[bool, str]:
    if not shutil.which(adb_bin):
        return False, f"{adb_bin} not on PATH"
    try:
        out = subprocess.run([adb_bin, "devices"], check=True, capture_output=True, text=True, timeout=5)
        return True, out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _check_path(p: Path) -> tuple[bool, str]:
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True, str(p)
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot create: {exc}"


def run_doctor() -> bool:
    s = get_settings()
    console = Console()
    table = Table(title="eat doctor", show_lines=False)
    for c in ("check", "ok", "detail"):
        table.add_column(c)

    rows: list[tuple[str, tuple[bool, str]]] = [
        ("repo root", (True, str(repo_root()))),
        ("configs/", _check_path(configs_dir())),
        ("projects/", _check_path(projects_dir())),
        ("artifacts/", _check_path(artifacts_dir())),
        ("runs/", _check_path(runs_dir())),
        ("cache/", _check_path(cache_dir())),
        ("redis", _check_redis(s.redis_url)),
        ("nvidia-smi", _check_gpu()),
        ("torch/cuda", _check_torch_cuda()),
        ("adb", _check_adb(s.adb_bin)),
        ("hf token", (bool(s.hf_token), "set" if s.hf_token else "missing (gated models need it)")),
        ("anthropic key", (bool(s.anthropic_api_key), "set" if s.anthropic_api_key else "missing (optional)")),
    ]
    all_ok = True
    for label, (ok, detail) in rows:
        table.add_row(label, "✓" if ok else "✗", detail)
        if not ok and label in {"configs/", "projects/", "artifacts/", "runs/", "cache/"}:
            all_ok = False
    console.print(table)
    return all_ok
