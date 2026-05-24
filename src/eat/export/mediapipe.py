"""MediaPipe `.task` export via Google AI Edge tools.

The exact toolchain is moving — ai-edge-torch and the GenAI Task bundler are
both under active development. We provide the shell here; if the optional
tools are not installed, we raise an actionable error explaining which package
to install for which model family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir

log = get_logger("export.mediapipe")


def export(run_id: str, params: dict[str, Any] | None = None) -> Path:
    params = params or {}
    out_dir = run_artifacts_dir(run_id)
    out = out_dir / "model.task"
    merged = out_dir / "merged"
    if not merged.exists():
        from eat.export import safetensors_merge
        safetensors_merge.export(run_id)

    try:
        # Real path: ai-edge-torch's GenAI conversion -> .task bundle.
        # The exact API depends on the model family — for Gemma 3n use
        # `ai_edge_torch.generative.examples.gemma3n` (subject to upstream change).
        import ai_edge_torch  # type: ignore  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "ai-edge-torch is required for MediaPipe export.\n"
            "  pip install ai-edge-torch ai-edge-torch-nightly mediapipe\n"
            "Then re-run. See https://ai.google.dev/edge/litert for current docs."
        ) from e

    raise NotImplementedError(
        "MediaPipe .task export wiring varies per model family; the upstream API "
        "is unstable as of 2026-05. Once Gemma 3n quantization is locked in "
        "ai-edge-torch, call into `ai_edge_torch.generative.examples.gemma3n` "
        "here to emit %s." % out
    )
