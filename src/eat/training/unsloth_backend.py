"""Unsloth fast-path backend (text-only).

Unsloth gives ~2x throughput and ~50% lower VRAM, but its multimodal support
trails the HF baseline. We use it only when the recipe is text-only AND the
optional `unsloth` extra is installed. Falls through to `hf_trl` otherwise.
"""

from __future__ import annotations

import json
from typing import IO, Any

from eat.config import get_settings
from eat.data.loaders import iter_jsonl
from eat.logging_setup import get_logger
from eat.paths import run_artifacts_dir
from eat.registry import datasets as datasets_reg
from eat.registry import models as models_reg
from eat.registry import projects as projects_reg
from eat.types import RecipeEntry

log = get_logger("training.unsloth")


def train(*, run_id: str, project: str, recipe: RecipeEntry, log_file: IO[str]) -> dict[str, Any]:
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError:
        log.warning("unsloth_missing_falling_back_to_hf_trl")
        from eat.training import hf_trl
        return hf_trl.train(run_id=run_id, project=project, recipe=recipe, log_file=log_file)

    s = get_settings()
    model_entry = models_reg.get(recipe.base_model)
    proj = projects_reg.get(project)
    out_dir = run_artifacts_dir(run_id)
    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    log_file.write(f"[unsloth] base={model_entry.hf_id}\n"); log_file.flush()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_entry.hf_id,
        max_seq_length=recipe.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        token=s.hf_token,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=recipe.lora_r,
        target_modules=list(recipe.lora_targets),
        lora_alpha=recipe.lora_alpha,
        lora_dropout=recipe.lora_dropout,
        bias="none",
        use_gradient_checkpointing=recipe.gradient_checkpointing,
        random_state=42,
    )

    from datasets import Dataset  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    items: list[dict[str, Any]] = []
    for ds_name in proj.datasets:
        ds = datasets_reg.get(ds_name)
        if ds.name == proj.eval_set:
            continue
        for r in iter_jsonl(__import__("pathlib").Path(ds.location)):
            if r.get("messages"):
                items.append({"messages": r["messages"]})
    if not items:
        raise RuntimeError("no training data for unsloth backend")

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
        logging_steps=10,
        save_steps=recipe.save_steps,
        save_total_limit=2,
        packing=False,
        report_to=["tensorboard"],
    )
    trainer = SFTTrainer(model=model, args=args,
                         train_dataset=Dataset.from_list(items),
                         tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    summary = {"metrics": {}, "artifacts": {"adapter": str(adapter_dir)}}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
