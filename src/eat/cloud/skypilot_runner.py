"""Launch a training job on the cloud via SkyPilot.

Shells out to the `sky` CLI because SkyPilot's Python API is heavy and
async-friendly only inside its own runtime. The orchestrator's cloud queue
calls into `launch()` which `sky launch -c <cluster>` and returns the cluster
name; teardown is the operator's job (or set `--down` to auto-terminate).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from eat.config import get_settings
from eat.logging_setup import get_logger
from eat.paths import cloud_dir

log = get_logger("cloud.sky")


def _sky() -> str:
    bin_ = shutil.which("sky")
    if not bin_:
        raise RuntimeError("sky CLI not on PATH; `pip install 'skypilot[runpod,lambda,aws]'`")
    return bin_


def launch(project: str, recipe: str, *, task: str = "train_qlora",
           cluster: str | None = None, down: bool = True) -> str:
    s = get_settings()
    task_yaml = cloud_dir() / "skypilot" / "tasks" / f"{task}.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(task_yaml)
    cluster = cluster or f"eat-{project[:8]}-{int(time.time()) % 10_000:04d}"
    env = os.environ.copy()
    env["PROJECT"] = project
    env["RECIPE"] = recipe
    args = [_sky(), "launch", "-c", cluster, "-y",
            "--env", f"PROJECT={project}", "--env", f"RECIPE={recipe}",
            "--env", f"HUGGING_FACE_HUB_TOKEN={s.hf_token or ''}",
            "--env", f"EAT_CLOUD_TRAIN_MAX_USD={s.cloud_train_max_usd}",
            "--env", f"EAT_CLOUD_TRAIN_MAX_HOURS={s.cloud_train_max_hours}",
            "--cloud", s.skypilot_default_cloud,
            str(task_yaml)]
    if down:
        args.append("--down")
    log.info("sky_launch", cluster=cluster, project=project, recipe=recipe)
    subprocess.run(args, check=True, env=env)
    return cluster


def teardown(cluster: str) -> None:
    subprocess.run([_sky(), "down", "-y", cluster], check=False)
