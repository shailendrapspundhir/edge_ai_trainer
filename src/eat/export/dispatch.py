"""Map a target entry to the right export module."""

from __future__ import annotations

from pathlib import Path

from eat.logging_setup import get_logger
from eat.registry import targets as targets_reg

log = get_logger("export.dispatch")


def export(run_id: str, target: str) -> Path:
    entry = targets_reg.get(target)
    if entry.kind == "gguf":
        from eat.export import gguf
        return gguf.export(run_id, quant=entry.quantization or "Q4_K_M")
    if entry.kind == "mediapipe":
        from eat.export import mediapipe
        return mediapipe.export(run_id, params=entry.params)
    if entry.kind == "onnx":
        from eat.export import onnx_export
        return onnx_export.export(run_id, params=entry.params)
    if entry.kind == "container":
        # Container builds happen in eat.cloud.containers; this returns the
        # placeholder path that the cloud_build job will overwrite.
        from eat.cloud import containers
        image = containers.build(run_id, target=target)
        return Path(image)  # not a filesystem path, but uniform return type
    if entry.kind == "safetensors":
        from eat.export import safetensors_merge
        return safetensors_merge.export(run_id)
    raise ValueError(f"unknown target kind: {entry.kind}")
