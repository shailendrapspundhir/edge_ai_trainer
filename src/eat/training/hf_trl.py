"""HuggingFace transformers + PEFT + TRL QLoRA backend.

Lazy-imports torch et al. so the orchestrator (which only plans + enqueues)
never has to load them. The worker process running on the GPU queue does.

Behaviour:
- 4-bit NF4 base via bitsandbytes; LoRA on top.
- SFTTrainer for chat-formatted data (`messages` field).
- Vision/audio encoders frozen (text-only path) — multimodal LoRA is gated
  behind `recipe.freeze_vision=false` and will go through `transformers`
  AutoModelForVision2Seq once Gemma 3n image support is mature there.
- Writes a merged adapter to `artifacts/<run_id>/adapter/`, and (best-effort)
  a merged-and-unquantised model to `artifacts/<run_id>/merged/` for export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from eat.config import get_settings
from eat.data.loaders import iter_jsonl
from eat.logging_setup import get_logger
from eat.paths import repo_root, run_artifacts_dir
from eat.registry import datasets as datasets_reg
from eat.registry import models as models_reg
from eat.registry import projects as projects_reg
from eat.types import RecipeEntry

log = get_logger("training.hf_trl")


def _format_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SFTTrainer accepts {"messages": [...]} directly."""
    out: list[dict[str, Any]] = []
    for r in records:
        msgs = r.get("messages")
        if msgs:
            out.append({"messages": msgs})
        elif "prompt" in r and "response" in r:
            out.append({"messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["response"]},
            ]})
    return out


def _resolve_ds_path(location: str) -> Path:
    p = Path(location)
    return p if p.is_absolute() else repo_root() / p


def _load_training_data(
    project: str,
    recipe: RecipeEntry | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    proj = projects_reg.get(project)
    train_ds_names = list(
        (getattr(recipe, "datasets", None) if recipe else None)
        or [d for d in (proj.datasets or []) if d != proj.eval_set]
    )
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for ds_name in train_ds_names:
        if ds_name == proj.eval_set:
            continue
        ds = datasets_reg.get(ds_name)
        train_records.extend(iter_jsonl(_resolve_ds_path(ds.location)))
    if proj.eval_set:
        ds = datasets_reg.get(proj.eval_set)
        eval_records.extend(iter_jsonl(_resolve_ds_path(ds.location)))
    return _format_examples(train_records), _format_examples(eval_records)


def train(*, run_id: str, project: str, recipe: RecipeEntry,
          log_file: IO[str]) -> dict[str, Any]:
    s = get_settings()
    model_entry = models_reg.get(recipe.base_model)

    log_file.write(f"loading base model: {model_entry.hf_id}\n"); log_file.flush()

    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer  # type: ignore

    out_dir = run_artifacts_dir(run_id)
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"
    adapter_dir.mkdir(exist_ok=True, parents=True)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=recipe.quantization or "nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_entry.hf_id, token=s.hf_token, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_entry.hf_id,
        quantization_config=quant_cfg,
        device_map="auto",
        dtype=torch.bfloat16,
        token=s.hf_token,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=recipe.gradient_checkpointing
    )

    lora_cfg = LoraConfig(
        r=recipe.lora_r,
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(recipe.lora_targets),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_items, eval_items = _load_training_data(project, recipe)
    log_file.write(f"data: train={len(train_items)} eval={len(eval_items)}\n"); log_file.flush()
    if not train_items:
        ds_list = recipe.datasets or projects_reg.get(project).datasets or []
        raise RuntimeError(
            f"no training data found for project {project!r} (datasets={ds_list}). "
            "Populate the project's processed/*.jsonl files, or set recipe.datasets "
            "to a hf-source dataset (e.g. smoke_text) so prepare_data can fetch it."
        )
    train_ds = Dataset.from_list(train_items)
    eval_ds = Dataset.from_list(eval_items) if eval_items else None

    args = SFTConfig(
        output_dir=str(adapter_dir),
        per_device_train_batch_size=recipe.per_device_batch_size,
        gradient_accumulation_steps=recipe.grad_accum,
        num_train_epochs=float(recipe.epochs),
        learning_rate=recipe.learning_rate,
        lr_scheduler_type=recipe.lr_scheduler,
        warmup_ratio=recipe.warmup_ratio,
        optim=recipe.optimizer,
        max_seq_length=recipe.max_seq_len,
        bf16=True,
        gradient_checkpointing=recipe.gradient_checkpointing,
        eval_strategy=recipe.eval_strategy if eval_ds is not None else "no",
        eval_steps=recipe.eval_steps,
        save_strategy="steps",
        save_steps=recipe.save_steps,
        save_total_limit=2,
        logging_steps=10,
        report_to=["tensorboard"],
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    # Best-effort merge to fp16 safetensors for downstream export.
    metrics = {"train_loss": float(trainer.state.log_history[-1].get("train_loss", 0.0))} \
        if trainer.state.log_history else {}
    artifacts = {"adapter": str(adapter_dir)}

    try:
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(merged_dir)
        artifacts["merged"] = str(merged_dir)
    except Exception as exc:  # noqa: BLE001
        log_file.write(f"merge_and_unload failed (non-fatal): {exc}\n")

    summary = {"metrics": metrics, "artifacts": artifacts}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("train_done", run_id=run_id, **metrics)
    return summary
