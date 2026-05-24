"""Hit a deployed endpoint (OpenAI-compatible) and record perf metrics."""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

import httpx

from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir

log = get_logger("cloud.bench")


_PROMPTS = [
    "Hello, introduce yourself in one short sentence.",
    "List three high-protein, low-sugar snack ideas.",
    "Why should a diabetic be cautious with mango juice?",
    "Suggest a 4 pm snack for someone with PCOS.",
    "Which foods should someone on warfarin limit?",
]


def _load_endpoints(run_id: str) -> dict[str, str]:
    f = run_artifacts_dir(run_id) / "endpoints.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return {}


def bench(run_id: str, endpoint: str | None = None,
          model_name: str = "model", max_tokens: int = 128) -> dict[str, Any]:
    if endpoint is None:
        eps = _load_endpoints(run_id)
        if not eps:
            raise RuntimeError(f"no recorded endpoints for run {run_id}; deploy first")
        endpoint = next(iter(eps.values()))

    url = endpoint.rstrip("/") + "/v1/chat/completions"
    samples = []
    with httpx.Client(timeout=120) as client:
        for prompt in _PROMPTS:
            t0 = time.perf_counter()
            r = client.post(url, json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            })
            r.raise_for_status()
            total = time.perf_counter() - t0
            data = r.json()
            usage = data.get("usage", {})
            out_tokens = usage.get("completion_tokens") or max_tokens
            samples.append({
                "prompt": prompt,
                "wall_s": total,
                "out_tokens": out_tokens,
                "decode_tps": out_tokens / total if total > 0 else 0.0,
            })
    decode_vals = [s["decode_tps"] for s in samples]
    metrics = {
        "median_decode_tps": statistics.median(decode_vals) if decode_vals else 0.0,
        "mean_wall_s": statistics.mean(s["wall_s"] for s in samples) if samples else 0.0,
        "n_prompts": len(samples),
    }
    out = run_artifacts_dir(run_id) / "cloud_bench_report.json"
    out.write_text(json.dumps({
        "endpoint": endpoint, "metrics": metrics, "samples": samples,
    }, indent=2))
    return {"path": str(out), "metrics": metrics}
