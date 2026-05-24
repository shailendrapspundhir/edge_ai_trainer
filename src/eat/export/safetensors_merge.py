"""Merge a LoRA adapter into the base model and write safetensors.

This is the input for the GGUF conversion and the cloud serving containers.
If `merged/` already exists from training, this is a no-op.
"""

from __future__ import annotations

from pathlib import Path

from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir

log = get_logger("export.safetensors_merge")


def export(run_id: str) -> Path:
    out_dir = run_artifacts_dir(run_id)
    merged = out_dir / "merged"
    if (merged / "config.json").exists():
        log.info("merged_already_present", path=str(merged))
        return merged

    adapter = out_dir / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"no adapter found at {adapter}. Train the run first or run "
            f"`eat run submit ...`")

    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    from eat.config import get_settings

    s = get_settings()
    # Read base model id from adapter_config
    import json
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    base_id = cfg.get("base_model_name_or_path")
    log.info("merging_adapter", base=base_id, adapter=str(adapter))

    tok = AutoTokenizer.from_pretrained(base_id, token=s.hf_token, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, token=s.hf_token,
        trust_remote_code=True, device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    merged_model = model.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged, safe_serialization=True, max_shard_size="2GB")
    tok.save_pretrained(merged)
    return merged
