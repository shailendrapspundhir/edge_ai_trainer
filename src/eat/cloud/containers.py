"""Build and tag container images for cloud serving targets.

Uses the local Docker daemon via `subprocess` (the `docker` Python SDK is
optional and adds little here). The output is a tagged image name; pushing
to a registry is a separate step (so we don't push half-built images).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from eat.config import get_settings
from eat.logging_setup import get_logger
from eat.paths import cloud_dir, repo_root, run_artifacts_dir
from eat.registry import targets as targets_reg

log = get_logger("cloud.containers")


def _docker() -> str:
    bin_ = shutil.which("docker")
    if not bin_:
        raise RuntimeError("docker CLI not found on PATH; install Docker first")
    return bin_


def _dockerfile_for(target_entry) -> Path:
    if target_entry.runtime == "vllm":
        return cloud_dir() / "docker" / "vllm.Dockerfile"
    if target_entry.runtime in ("llama.cpp-server", "llamacpp"):
        return cloud_dir() / "docker" / "llamacpp.Dockerfile"
    raise ValueError(f"no Dockerfile for runtime {target_entry.runtime!r}")


def _tag_for(run_id: str, target_name: str) -> str:
    registry = get_settings().container_registry.rstrip("/")
    return f"{registry}:{target_name}-{run_id}"


def build(run_id: str, target: str, push: bool = False) -> str:
    entry = targets_reg.get(target)
    df = _dockerfile_for(entry)
    if not df.exists():
        raise FileNotFoundError(f"missing Dockerfile: {df}")
    tag = _tag_for(run_id, target)

    # Prepare build context: copy the model dir into a build-context staging area
    # so the `COPY` in the Dockerfile works.
    out = run_artifacts_dir(run_id)
    if entry.runtime == "vllm":
        model_dir = out / "merged"
        if not model_dir.exists():
            from eat.export import safetensors_merge
            safetensors_merge.export(run_id)
        bake = "true"
        build_args = ["--build-arg", f"BAKE_MODEL={bake}",
                      "--build-arg", f"BUILD_MODEL_DIR={model_dir.relative_to(repo_root())}"]
    elif entry.runtime in ("llama.cpp-server", "llamacpp"):
        gguf = next(out.glob("model.Q4_K_M.gguf"), None) or next(out.glob("*.gguf"), None)
        if not gguf:
            raise FileNotFoundError(f"no GGUF in {out}; run `eat export gguf {run_id}` first")
        build_args = ["--build-arg", "BAKE_MODEL=true",
                      "--build-arg", f"GGUF_FILE={gguf.name}",
                      "--build-arg", f"BUILD_MODEL_DIR={out.relative_to(repo_root())}"]
    else:
        build_args = []

    cmd = [_docker(), "build", "-f", str(df), "-t", tag, *build_args, str(repo_root())]
    log.info("docker_build", cmd=" ".join(cmd))
    subprocess.run(cmd, check=True)

    if push:
        subprocess.run([_docker(), "push", tag], check=True)
    return tag
