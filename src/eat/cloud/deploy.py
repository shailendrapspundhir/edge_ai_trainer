"""Provider-specific deploy/destroy drivers.

Each provider is wired through a small dispatcher; adding a new one means a
new `_deploy_<provider>` function and a row in the map. Returns the public
endpoint URL on success.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from eat.config import get_settings
from eat.logging_setup import get_logger
from eat.paths import cloud_dir, repo_root, run_artifacts_dir
from eat.registry import targets as targets_reg

log = get_logger("cloud.deploy")


def deploy(run_id: str, target: str) -> str:
    entry = targets_reg.get(target)
    platform = (entry.platform or "").lower()
    fn = {
        "cloud:modal": _deploy_modal,
        "cloud:vllm": _deploy_generic_docker,
        "cloud:llamacpp": _deploy_generic_docker,
    }.get(platform)
    if fn is None:
        raise ValueError(f"no deploy driver for platform {platform!r}")
    endpoint = fn(run_id, target, entry)
    _record_endpoint(run_id, target, endpoint)
    return endpoint


def destroy(deployment_id: str, target: str) -> None:
    entry = targets_reg.get(target)
    platform = (entry.platform or "").lower()
    if platform == "cloud:modal":
        subprocess.run(["modal", "app", "stop", deployment_id], check=False)
        return
    log.warning("destroy_not_implemented", platform=platform, deployment_id=deployment_id)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


def _deploy_modal(run_id: str, target: str, entry) -> str:
    if not shutil.which("modal"):
        raise RuntimeError("modal CLI not on PATH; `pip install modal` and run `modal token new`")
    app_py = cloud_dir() / "deploy" / "modal_app.py"
    if not app_py.exists():
        raise FileNotFoundError(f"{app_py} missing")
    env = os.environ.copy()
    env.update({
        "EAT_RUN_ID": run_id,
        "EAT_TARGET": target,
        "EAT_MODAL_RUNTIME": "vllm" if entry.runtime == "vllm" else "llamacpp",
    })
    log.info("modal_deploy", target=target, run_id=run_id)
    result = subprocess.run(
        ["modal", "deploy", str(app_py)],
        env=env, capture_output=True, text=True, check=True,
    )
    # Parse a URL out of modal's stdout (it usually prints a hosted endpoint).
    for line in result.stdout.splitlines():
        if "https://" in line:
            return line.strip().split()[-1]
    return "(deployed; check `modal app list`)"


def _deploy_generic_docker(run_id: str, target: str, entry) -> str:
    """Just runs the image locally for hand-testing. Real prod deploys should
    go through a specific provider driver (modal, runpod, fly, k8s)."""
    image_tag = f"{get_settings().container_registry.rstrip('/')}:{target}-{run_id}"
    port = int(entry.params.get("port", 8000))
    name = f"eat-{run_id}-{target}"
    subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{port}:{port}", image_tag],
        check=True,
    )
    return f"http://127.0.0.1:{port}"


def _record_endpoint(run_id: str, target: str, endpoint: str) -> None:
    out = run_artifacts_dir(run_id) / "endpoints.json"
    data: dict = {}
    if out.exists():
        try:
            data = json.loads(out.read_text())
        except json.JSONDecodeError:
            data = {}
    data[target] = endpoint
    out.write_text(json.dumps(data, indent=2))
