"""Sweep engine — orchestrates N trials through the planner with ASHA scheduling.

The engine is designed to run *without* Redis/RQ. It executes trials
sequentially in-process (for local/CI use) or can be extended to dispatch
via the orchestrator queue.  Each trial is a standalone training run with
recipe overrides sampled from the search space.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from eat.logging_setup import get_logger
from eat.sweep.asha import ASHAScheduler
from eat.sweep.sampler import sample
from eat.sweep.types import (
    MultiObjective,
    SearchSpace,
    SweepRecord,
    SweepStatus,
    TrialRecord,
    TrialStatus,
)

log = get_logger("sweep.engine")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _apply_overrides(base_recipe_dict: dict[str, Any], hp: dict[str, Any]) -> dict[str, Any]:
    """Overlay sampled hyperparams onto the base recipe dict."""
    merged = dict(base_recipe_dict)
    for k, v in hp.items():
        merged[k] = v
    return merged


# ---------------------------------------------------------------------------
# In-process local trainer for a single trial
# ---------------------------------------------------------------------------


def _run_trial_local(
    *,
    trial: TrialRecord,
    recipe_overrides: dict[str, Any],
    data_path: str,
    model_path: str,
    output_dir: str,
    epochs: float,
    log_file: IO[str],
) -> dict[str, float]:
    """Execute one trial locally and return metrics.

    This mirrors the e2e pipeline_steps train logic, using SFTTrainer
    directly to avoid requiring Redis/orchestrator.
    """
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore
    import time

    log_file.write(f"trial {trial.id}: loading model {model_path}\n")
    log_file.flush()

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        local_files_only=True,
    )

    lora_r = int(recipe_overrides.get("lora_r", 16))
    lora_alpha = int(recipe_overrides.get("lora_alpha", 32))
    lora_dropout = float(recipe_overrides.get("lora_dropout", 0.05))
    lora_targets = recipe_overrides.get(
        "lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_cfg)

    records = [json.loads(line) for line in Path(data_path).read_text().splitlines() if line.strip()]
    train_ds = Dataset.from_list(records)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lr = float(recipe_overrides.get("learning_rate", 3e-4))
    batch_size = int(recipe_overrides.get("per_device_batch_size", 2))
    grad_accum = int(recipe_overrides.get("grad_accum", 1))
    lr_scheduler = recipe_overrides.get("lr_scheduler", "cosine")
    warmup_ratio = float(recipe_overrides.get("warmup_ratio", 0.05))
    max_seq_len = int(recipe_overrides.get("max_seq_len", 256))
    optimizer = recipe_overrides.get(
        "optimizer", "paged_adamw_8bit" if use_cuda else "adamw_torch",
    )

    sft_args = SFTConfig(
        output_dir=str(out_path),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup_ratio,
        optim=optimizer,
        max_length=max_seq_len,
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=False,
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
        use_cpu=not use_cuda,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    train_loss = result.metrics.get("train_loss", float("inf"))
    total_steps = result.global_step
    # rough throughput: total tokens / elapsed
    tokens_processed = sum(
        len(tokenizer.encode(json.dumps(r.get("messages", ""))))
        for r in records
    ) * epochs
    tokens_per_sec = tokens_processed / max(elapsed, 0.01)

    # peak memory
    if use_cuda:
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        try:
            import resource
            peak_mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            peak_mem_mb = 0.0

    metrics = {
        "train_loss": train_loss,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_mb": peak_mem_mb,
        "train_runtime_s": elapsed,
        "total_steps": float(total_steps),
    }

    log_file.write(
        f"trial {trial.id} done: loss={train_loss:.4f} "
        f"tok/s={tokens_per_sec:.1f} mem={peak_mem_mb:.0f}MB "
        f"runtime={elapsed:.1f}s\n"
    )
    log_file.flush()
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    project: str,
    recipe: str,
    search_space: SearchSpace,
    objectives: MultiObjective,
    max_trials: int,
    data_path: str,
    model_path: str,
    output_dir: str,
    asha_max_rung: int = 3,
    asha_reduction_factor: int = 3,
    asha_min_epochs: float = 1.0,
    log_file: IO[str] | None = None,
    seed: int | None = None,
) -> SweepRecord:
    """Execute a full sweep: sample N trials, ASHA-schedule them, return best.

    This is the standalone local entry point. Each trial trains in-process.
    """
    import sys
    _log = log_file or sys.stdout

    sweep_id = _new_id("sweep")
    sweep = SweepRecord(
        id=sweep_id,
        project=project,
        recipe=recipe,
        max_trials=max_trials,
        search_space=search_space,
        objectives=objectives,
        status=SweepStatus.RUNNING,
        asha_max_rung=asha_max_rung,
        asha_reduction_factor=asha_reduction_factor,
        asha_min_epochs=asha_min_epochs,
    )

    scheduler = ASHAScheduler(
        max_rung=asha_max_rung,
        reduction_factor=asha_reduction_factor,
        min_epochs=asha_min_epochs,
        objectives=objectives,
    )

    # --- Sample all trials upfront ---
    trials: list[TrialRecord] = []
    for i in range(max_trials):
        hp = sample(search_space, seed=(seed + i) if seed is not None else None)
        trial = TrialRecord(
            id=_new_id("trial"),
            sweep_id=sweep_id,
            params=hp,
        )
        trials.append(trial)
        sweep.trial_ids.append(trial.id)

    _log.write(f"sweep {sweep_id}: {max_trials} trials, {len(search_space.params)} HP axes\n")
    _log.write(f"  ASHA: max_rung={asha_max_rung} reduction={asha_reduction_factor} "
               f"min_epochs={asha_min_epochs}\n")
    _log.flush()

    # --- Execute rung-by-rung with ASHA ---
    active = list(trials)
    for rung in range(asha_max_rung + 1):
        if not active:
            break
        epochs = scheduler.epochs_for_rung(rung)
        _log.write(f"\n=== Rung {rung}: {len(active)} trial(s), {epochs} epoch(s) ===\n")
        _log.flush()

        for trial in active:
            trial.rung = rung
            trial.status = TrialStatus.RUNNING
            trial_dir = str(Path(output_dir) / sweep_id / trial.id / f"rung{rung}")

            try:
                metrics = _run_trial_local(
                    trial=trial,
                    recipe_overrides=trial.params,
                    data_path=data_path,
                    model_path=model_path,
                    output_dir=trial_dir,
                    epochs=epochs,
                    log_file=_log,
                )
                trial.metrics = metrics
                trial.epochs_completed = epochs
                trial.status = TrialStatus.COMPLETED
            except Exception as exc:
                _log.write(f"trial {trial.id} FAILED: {exc}\n")
                _log.flush()
                trial.status = TrialStatus.FAILED
                trial.finished_at = datetime.utcnow()

        # --- ASHA promotion decisions ---
        next_active: list[TrialRecord] = []
        for trial in active:
            decision = scheduler.decide(trial, trials)
            if decision == "promote":
                next_active.append(trial)
                _log.write(f"  trial {trial.id}: PROMOTED (score={scheduler.scalarise(trial.metrics):.4f})\n")
            else:
                if trial.status == TrialStatus.COMPLETED:
                    trial.status = TrialStatus.EARLY_STOPPED
                trial.finished_at = datetime.utcnow()
                _log.write(f"  trial {trial.id}: STOPPED  (score={scheduler.scalarise(trial.metrics):.4f})\n")
        _log.flush()
        active = next_active

    # --- Mark remaining active trials as completed ---
    for trial in active:
        trial.finished_at = datetime.utcnow()

    # --- Pick best trial ---
    completed = [t for t in trials if t.status in (
        TrialStatus.COMPLETED, TrialStatus.EARLY_STOPPED,
    )]
    if completed:
        best = max(completed, key=lambda t: scheduler.scalarise(t.metrics))
        sweep.best_trial_id = best.id
    sweep.status = SweepStatus.COMPLETED
    sweep.finished_at = datetime.utcnow()

    # --- Write sweep summary ---
    summary_path = Path(output_dir) / sweep_id / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "sweep_id": sweep.id,
        "status": sweep.status.value,
        "max_trials": max_trials,
        "best_trial_id": sweep.best_trial_id,
        "trials": [
            {
                "id": t.id,
                "status": t.status.value,
                "rung": t.rung,
                "params": t.params,
                "metrics": t.metrics,
                "epochs_completed": t.epochs_completed,
            }
            for t in trials
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    _log.write(f"\nsweep {sweep_id} complete: "
               f"{len(completed)} finished, best={sweep.best_trial_id}\n")
    if sweep.best_trial_id:
        best_t = next(t for t in trials if t.id == sweep.best_trial_id)
        _log.write(f"  best params: {json.dumps(best_t.params, default=str)}\n")
        _log.write(f"  best metrics: {json.dumps(best_t.metrics, default=str)}\n")
    _log.flush()

    return sweep
