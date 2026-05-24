"""Performance eval — tokens/sec, TTFT, peak RAM, file size.

The runtime determines how we measure:

    llama.cpp     -> shell out to `llama-cli -m model.gguf -n 128 -p ... -t N`,
                     parse the timing lines.
    mediapipe     -> deferred to the Android runner (see eat.android.bench).
    vllm/ort/etc  -> launched via the corresponding eat.cloud or eat.export
                     modules and timed with httpx / direct ORT session.

For v0 we implement the llama.cpp path concretely and stub the others so
they emit empty reports rather than failing.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir

log = get_logger("eval.perf")


_PROMPTS = [
    "Hello, tell me about yourself in one sentence.",
    "What are three high-protein, low-sugar snack ideas?",
    "Explain why a diabetic should be cautious with mango.",
    "Suggest a 4 pm snack for a pregnant woman with gestational diabetes.",
    "If I'm on warfarin, which foods should I avoid?",
]


def _llama_cli() -> str | None:
    return shutil.which("llama-cli") or shutil.which("main")


def _gguf_path(run_id: str, quant: str) -> Path | None:
    out_dir = run_artifacts_dir(run_id)
    for p in out_dir.glob(f"*.{quant}.gguf"):
        return p
    for p in out_dir.glob("*.gguf"):
        return p
    return None


def _bench_llama_cpp(model_path: Path, prompts: list[str], n_tokens: int = 128) -> dict[str, Any]:
    cli = _llama_cli()
    if not cli:
        log.warning("llama_cpp_not_found")
        return {"metrics": {}, "samples": [],
                "warning": "llama-cli not on PATH; install llama.cpp"}

    samples = []
    for prompt in prompts:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [cli, "-m", str(model_path), "-n", str(n_tokens),
             "-p", prompt, "--no-display-prompt", "-t", "8"],
            capture_output=True, text=True, check=False, timeout=180,
        )
        elapsed = time.perf_counter() - t0
        out = proc.stdout + proc.stderr
        tps = _parse_tps(out)
        ttft = _parse_ttft(out)
        samples.append({"prompt": prompt, "wall_s": elapsed,
                        "decode_tps": tps, "ttft_ms": ttft})
    decode_tps_vals = [s["decode_tps"] for s in samples if s.get("decode_tps")]
    ttft_vals = [s["ttft_ms"] for s in samples if s.get("ttft_ms")]
    metrics = {
        "median_decode_tps": statistics.median(decode_tps_vals) if decode_tps_vals else 0.0,
        "p95_ttft_ms": _percentile(ttft_vals, 95) if ttft_vals else 0.0,
        "file_size_mb": model_path.stat().st_size / (1024 * 1024),
    }
    return {"metrics": metrics, "samples": samples}


def _parse_tps(text: str) -> float | None:
    # llama.cpp prints lines like: "llama_print_timings:        eval time = ... ( ... tokens per second)"
    m = re.search(r"eval\s+time\s*=.*?\(\s*([0-9.]+)\s*tokens?\s*per\s*second\s*\)", text, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9.]+)\s*tokens?/s", text)
    return float(m.group(1)) if m else None


def _parse_ttft(text: str) -> float | None:
    m = re.search(r"prompt\s+eval\s+time\s*=\s*([0-9.]+)\s*ms", text, re.I)
    return float(m.group(1)) if m else None


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100) * (len(s) - 1)))))
    return s[k]


def evaluate(run_id: str, runtime: str = "llama.cpp", quant: str = "Q4_K_M") -> dict[str, Any]:
    out_dir = run_artifacts_dir(run_id)
    if runtime == "llama.cpp":
        model = _gguf_path(run_id, quant)
        if not model:
            report = {"runtime": runtime, "metrics": {},
                      "warning": f"no GGUF for run_id={run_id} (quant={quant}); run export first"}
        else:
            result = _bench_llama_cpp(model, _PROMPTS)
            report = {"runtime": runtime, "quant": quant, "model_path": str(model),
                      "metrics": result["metrics"], "samples": result["samples"]}
    else:
        report = {"runtime": runtime, "metrics": {},
                  "warning": f"runtime {runtime!r} not yet implemented in eat.eval.perf"}

    out = out_dir / f"perf_report.{runtime.replace('.', '_')}.json"
    out.write_text(json.dumps(report, indent=2))
    return {"path": str(out), "metrics": report.get("metrics", {})}
