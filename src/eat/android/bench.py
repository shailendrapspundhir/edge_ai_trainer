"""Drive the on-device benchmark APK and pull metrics.

Flow:
    1. Resolve the model artifact for `target` under artifacts/<run_id>/.
       - mediapipe_litert -> model.task
       - gguf_q*           -> model.<quant>.gguf
    2. adb push model + prompts JSON to /data/local/tmp/eat/<run_id>/
    3. adb start MainActivity --wait with the right extras
    4. adb pull the result JSON; parse and return.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eat.android import adb_runner
from eat.config import get_settings
from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir
from eat.registry import targets as targets_reg

log = get_logger("android.bench")


_PROMPTS = {
    "max_tokens": 128,
    "prompts": [
        {"id": "p1", "text": "Hello, introduce yourself in one short sentence."},
        {"id": "p2", "text": "Three high-protein, low-sugar snack ideas?"},
        {"id": "p3", "text": "Explain why a diabetic should be cautious with mango."},
        {"id": "p4", "text": "Suggest a 4 pm snack for someone with PCOS."},
        {"id": "p5", "text": "On warfarin, which foods should I limit?"},
    ],
}


def _model_artifact_for(run_id: str, target_entry) -> Path:
    out = run_artifacts_dir(run_id)
    if target_entry.kind == "mediapipe":
        cand = out / "model.task"
        if not cand.exists():
            raise FileNotFoundError(f"{cand} missing — run `eat export mediapipe {run_id}` first")
        return cand
    if target_entry.kind == "gguf":
        q = target_entry.quantization or "Q4_K_M"
        cand = out / f"model.{q}.gguf"
        if not cand.exists():
            raise FileNotFoundError(f"{cand} missing — run `eat export gguf {run_id} --quant {q}` first")
        return cand
    raise ValueError(f"target kind {target_entry.kind} not deployable to Android via this runner")


def run_bench(
    run_id: str,
    target: str = "mediapipe_litert",
    device_serial: str | None = None,
    prompts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    s = get_settings()
    entry = targets_reg.get(target)
    serial = device_serial or s.android_device_serial or adb_runner.first_serial()

    model_path = _model_artifact_for(run_id, entry)
    remote_dir = f"{s.android_remote_dir.rstrip('/')}/{run_id}"
    adb_runner.remote_mkdir(remote_dir, serial=serial)
    remote_model = f"{remote_dir}/{model_path.name}"
    remote_prompts = f"{remote_dir}/prompts.json"
    remote_output = f"{remote_dir}/result.json"

    log.info("android_push_model", local=str(model_path), remote=remote_model, serial=serial)
    adb_runner.push(model_path, remote_model, serial=serial)

    prompts_payload = prompts or _PROMPTS
    local_prompts = run_artifacts_dir(run_id) / "android_prompts.json"
    local_prompts.write_text(json.dumps(prompts_payload, ensure_ascii=False))
    adb_runner.push(local_prompts, remote_prompts, serial=serial)

    runtime = "mediapipe" if entry.kind == "mediapipe" else "llamacpp"
    log.info("android_run", target=target, runtime=runtime)
    adb_runner.start_intent(
        component=f"{s.android_app_id}/.MainActivity",
        action="com.eat.bench.RUN_BENCH",
        extras={
            "model_path": remote_model,
            "runtime": runtime,
            "prompts_path": remote_prompts,
            "output_path": remote_output,
            "warmup": "1",
        },
        serial=serial,
        wait=True,
        timeout=s.android_bench_timeout_s,
    )

    local_out = run_artifacts_dir(run_id) / f"android_report.{target}.json"
    adb_runner.pull(remote_output, local_out, serial=serial)
    data = json.loads(local_out.read_text())
    summary = data.get("summary", {})
    return {
        "path": str(local_out),
        "target": target,
        "device": serial,
        "metrics": {
            "median_decode_tps": float(summary.get("median_decode_tokens_per_s", 0.0)),
            "p95_ttft_ms": float(summary.get("p95_ttft_ms", 0.0)),
            "peak_rss_mb": float(summary.get("peak_rss_mb", 0.0)),
        },
    }
