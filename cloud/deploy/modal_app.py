"""Modal deployment wrapper for an EAT-trained model.

This file exposes a single Modal app that can serve either:

  * a vLLM OpenAI-compatible GPU server (EAT_MODAL_RUNTIME=vllm, default), or
  * a llama.cpp HTTP server (EAT_MODAL_RUNTIME=llamacpp) for CPU/cheap GPU use.

It reads the model from a Modal Volume (default name: ``eat-models``) mounted at
``/models``. Use :func:`seed_volume` once locally before deploying to upload the
exported model into the volume.

Environment variables (set in your shell before ``modal deploy``):

================================  ==========================================
EAT_RUN_ID                        Run id of the EAT export to serve. Used
                                  only for labelling / glob discovery.
EAT_TARGET                        e.g. ``cloud_vllm_gpu`` or
                                  ``cloud_llamacpp_cpu`` — informational.
EAT_MODAL_RUNTIME                 ``vllm`` (default) | ``llamacpp``.
EAT_MODAL_VOLUME                  Modal volume name. Default ``eat-models``.
EAT_MODEL_GLOB                    Relative path inside the volume that holds
                                  the model. Default ``model/``.
================================  ==========================================

Usage::

    # 1. Seed the volume from a local export (one-off):
    modal run cloud/deploy/modal_app.py::seed_volume \\
        --local-path ./artifacts/runs/<run_id>/export/merged

    # 2. Deploy:
    EAT_MODAL_RUNTIME=vllm \\
    modal deploy cloud/deploy/modal_app.py

    # 3. The web endpoint URL is printed by Modal on deploy. Hit /v1/models or
    #    /v1/chat/completions (vLLM) or /completion (llama.cpp).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config from environment.
# ---------------------------------------------------------------------------
EAT_RUN_ID = os.environ.get("EAT_RUN_ID", "unknown")
EAT_TARGET = os.environ.get("EAT_TARGET", "cloud_modal")
EAT_MODAL_RUNTIME = os.environ.get("EAT_MODAL_RUNTIME", "vllm").lower()
EAT_MODAL_VOLUME = os.environ.get("EAT_MODAL_VOLUME", "eat-models")
EAT_MODEL_GLOB = os.environ.get("EAT_MODEL_GLOB", "model/")

if EAT_MODAL_RUNTIME not in {"vllm", "llamacpp"}:
    raise ValueError(
        f"EAT_MODAL_RUNTIME must be 'vllm' or 'llamacpp', got {EAT_MODAL_RUNTIME!r}"
    )

MODEL_MOUNT = "/models"
MODEL_DIR = f"{MODEL_MOUNT}/{EAT_MODEL_GLOB.rstrip('/')}"

# ---------------------------------------------------------------------------
# Images — both defined, only the active one is used by `serve`.
# ---------------------------------------------------------------------------
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .pip_install(
        "vllm==0.6.2",
        "fastapi",
        "huggingface_hub>=0.24",
    )
    .env(
        {
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "HF_HOME": "/root/.cache/hf",
        }
    )
)

llamacpp_image = (
    modal.Image.from_registry("ghcr.io/ggerganov/llama.cpp:server", add_python="3.11")
    .pip_install("fastapi", "httpx")
)

ACTIVE_IMAGE = vllm_image if EAT_MODAL_RUNTIME == "vllm" else llamacpp_image
ACTIVE_GPU = "L4" if EAT_MODAL_RUNTIME == "vllm" else None

# ---------------------------------------------------------------------------
# Modal app + volume.
# ---------------------------------------------------------------------------
app = modal.App(name=f"eat-{EAT_RUN_ID}-{EAT_MODAL_RUNTIME}")
volume = modal.Volume.from_name(EAT_MODAL_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Volume seeding — run once before `modal deploy`.
# ---------------------------------------------------------------------------
@app.function(
    volumes={MODEL_MOUNT: volume},
    timeout=60 * 60,
)
def seed_volume(local_path: str, dest: str = "model") -> str:
    """Upload a local model directory into the Modal volume.

    Invoke locally::

        modal run cloud/deploy/modal_app.py::seed_volume \\
            --local-path ./artifacts/runs/<run_id>/export/merged

    `dest` is the subdirectory inside the volume (matches EAT_MODEL_GLOB).
    """
    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(f"local_path not found: {local_path}")

    target = Path(MODEL_MOUNT) / dest
    target.mkdir(parents=True, exist_ok=True)

    # Use rsync-like copy; modal volume commit happens automatically on return.
    for item in src.rglob("*"):
        if item.is_file():
            rel = item.relative_to(src)
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(item.read_bytes())

    volume.commit()
    return f"seeded {target} from {src}"


# ---------------------------------------------------------------------------
# Serve — the public web endpoint.
# ---------------------------------------------------------------------------
@app.function(
    image=ACTIVE_IMAGE,
    gpu=ACTIVE_GPU,
    volumes={MODEL_MOUNT: volume},
    container_idle_timeout=300,
    allow_concurrent_inputs=16,
    timeout=60 * 60,
)
@modal.asgi_app()
def serve():
    """Expose the underlying inference server as a Modal ASGI app."""
    if EAT_MODAL_RUNTIME == "vllm":
        return _build_vllm_app()
    return _build_llamacpp_app()


def _build_vllm_app():
    """Spin up vLLM in-process and return its FastAPI app."""
    from vllm.entrypoints.openai.api_server import build_app  # type: ignore
    from vllm.entrypoints.openai.cli_args import make_arg_parser  # type: ignore

    parser = make_arg_parser()
    args = parser.parse_args(
        [
            "--model", MODEL_DIR,
            "--served-model-name", "model",
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--host", "0.0.0.0",
            "--port", "8000",
        ]
    )
    return build_app(args)


def _build_llamacpp_app():
    """Run llama-server as a subprocess and proxy via FastAPI."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    # Find the first .gguf file under MODEL_DIR.
    gguf_files = list(Path(MODEL_DIR).glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf model found under {MODEL_DIR}")
    gguf = str(gguf_files[0])

    # Boot llama-server on an internal port; FastAPI proxies to it.
    proc = subprocess.Popen(
        [
            "llama-server",
            "-m", gguf,
            "-c", "4096",
            "-t", "8",
            "--host", "127.0.0.1",
            "--port", "8081",
        ]
    )

    upstream = "http://127.0.0.1:8081"
    api = FastAPI()
    client = httpx.AsyncClient(base_url=upstream, timeout=None)

    @api.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(path: str, request: Request) -> Response:
        body = await request.body()
        upstream_resp = await client.request(
            request.method,
            f"/{path}",
            content=body,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            params=request.query_params,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=dict(upstream_resp.headers),
        )

    @api.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover
        await client.aclose()
        proc.terminate()

    return api


# ---------------------------------------------------------------------------
# Local entrypoint helper. `modal deploy <this-file>` will also work.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main() -> None:
    print(f"eat modal app: run_id={EAT_RUN_ID} target={EAT_TARGET} runtime={EAT_MODAL_RUNTIME}")
    print(f"  volume={EAT_MODAL_VOLUME} mount={MODEL_MOUNT} model_dir={MODEL_DIR}")
    print("Use `modal deploy cloud/deploy/modal_app.py` to deploy.")
