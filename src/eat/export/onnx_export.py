"""ONNX export via `optimum` — secondary path for non-Gemma models later."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir

log = get_logger("export.onnx")


def export(run_id: str, params: dict[str, Any] | None = None) -> Path:
    params = params or {}
    out_dir = run_artifacts_dir(run_id) / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = run_artifacts_dir(run_id) / "merged"
    if not merged.exists():
        from eat.export import safetensors_merge
        safetensors_merge.export(run_id)

    try:
        from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "ONNX export needs `pip install optimum[onnxruntime] onnx onnxruntime`."
        ) from e

    log.info("onnx_exporting", merged=str(merged), out=str(out_dir))
    tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
    ORTModelForCausalLM.from_pretrained(merged, export=True).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir
