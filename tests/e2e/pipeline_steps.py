#!/usr/bin/env python3
"""Pipeline step runner for E2E training tests.

Each subcommand corresponds to one stage of the training pipeline. The bash
test script (test_training_pipeline.sh) calls these one-by-one and checks
exit codes.

All model loading uses local_files_only=True to ensure zero HuggingFace
network requests. The model must be pre-cached in llm_foundation_models/.

Steps:
    validate-data       Verify the JSONL fixture is well-formed
    check-prereqs       Confirm torch / peft / trl / transformers are installed
    verify-model        Assert the local model directory has required files
    train               LoRA fine-tune with checkpointing
    verify-checkpoints  Assert checkpoint dirs were created
    verify-adapter      Assert final adapter artefacts exist
    verify-inference    Load base vs fine-tuned and compare outputs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU detection — MUST run before any `import torch` to be effective
# ---------------------------------------------------------------------------

_MIN_VRAM_MIB = 2048  # 2 GB


def _gpu_free_mib() -> int | None:
    """Query free VRAM via nvidia-smi (no torch import). Returns MiB or None."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split("\n")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _ensure_device_env() -> bool:
    """If GPU VRAM is insufficient, hide CUDA from torch. Returns use_cuda."""
    free = _gpu_free_mib()
    if free is not None and free < _MIN_VRAM_MIB:
        print(f"  VRAM: {free} MiB free — below {_MIN_VRAM_MIB} MiB threshold")
        print("  Forcing CPU mode (another process may be using the GPU)")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return False
    if free is not None:
        print(f"  VRAM: {free} MiB free — OK")
    return True  # let torch decide


# ---------------------------------------------------------------------------
# Step 1: validate-data
# ---------------------------------------------------------------------------

def validate_data(args: argparse.Namespace) -> int:
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"FAIL: data file not found: {data_path}")
        return 1

    records: list[dict] = []
    with data_path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"FAIL: invalid JSON on line {i}: {e}")
                return 1
            if "messages" not in record:
                print(f"FAIL: missing 'messages' key on line {i}")
                return 1
            msgs = record["messages"]
            if not isinstance(msgs, list) or len(msgs) < 2:
                print(f"FAIL: expected >= 2 messages on line {i}, got {len(msgs) if isinstance(msgs, list) else 'non-list'}")
                return 1
            records.append(record)

    if len(records) < args.min_records:
        print(f"FAIL: expected >= {args.min_records} records, got {len(records)}")
        return 1

    total_chars = sum(len(m["content"]) for r in records for m in r["messages"])
    print(f"PASS: {len(records)} records, {total_chars:,} total characters")
    return 0


# ---------------------------------------------------------------------------
# Step 2: check-prereqs
# ---------------------------------------------------------------------------

def check_prereqs(_args: argparse.Namespace) -> int:
    missing: list[str] = []
    for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"FAIL: missing packages: {', '.join(missing)}")
        print("  Install with: uv sync --extra train")
        return 1

    import torch  # type: ignore

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(f"PASS: all packages available  (PyTorch {torch.__version__}, device={device})")
    if cuda:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("  (no CUDA — training will use CPU / fp32, expect ~15-30 min)")
    return 0


# ---------------------------------------------------------------------------
# Step 3: verify-model (local — no HuggingFace requests)
# ---------------------------------------------------------------------------

def verify_model(args: argparse.Namespace) -> int:
    model_path = Path(args.model_path)
    print(f"Verifying local model: {model_path}")

    if not model_path.is_dir():
        print(f"FAIL: model directory not found: {model_path}")
        return 1

    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    missing = [f for f in required if not (model_path / f).exists()]
    has_weights = (
        (model_path / "model.safetensors").exists()
        or any(model_path.glob("model-*.safetensors"))
        or (model_path / "pytorch_model.bin").exists()
    )

    if missing:
        print(f"FAIL: missing required files: {', '.join(missing)}")
        return 1
    if not has_weights:
        print("FAIL: missing model weights (model.safetensors or pytorch_model.bin)")
        return 1

    files = sorted(f.name for f in model_path.iterdir() if f.is_file())
    total_bytes = sum(f.stat().st_size for f in model_path.iterdir() if f.is_file())
    print(f"PASS: local model verified ({total_bytes / 1e6:.0f} MB)")
    print(f"  files: {', '.join(files)}")
    return 0


# ---------------------------------------------------------------------------
# Step 4: train
# ---------------------------------------------------------------------------

def train_step(args: argparse.Namespace) -> int:
    _ensure_device_env()  # must run before importing torch

    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    adapter_dir = output_dir / "final_adapter"
    model_path = args.model_path
    epochs = args.epochs
    save_steps = args.save_steps

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if use_cuda else 'cpu'}")

    # ── tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model ──────────────────────────────────────────────────────────────
    print("Loading base model …")
    if use_cuda:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
            from peft import prepare_model_for_kbit_training  # type: ignore

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_cfg,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="eager",
            )
            model = prepare_model_for_kbit_training(model)
            print("  loaded in 4-bit NF4 (GPU)")
        except Exception as exc:
            print(f"  4-bit load failed ({exc}), falling back to fp16 GPU")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                )
                print("  loaded in fp16 (GPU)")
            except Exception as exc2:
                print(f"  fp16 GPU also failed ({exc2}), falling back to CPU")
                use_cuda = False
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                print("  loaded in fp32 (CPU)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("  loaded in fp32 (CPU)")

    # ── LoRA ───────────────────────────────────────────────────────────────
    print("Applying LoRA …")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── dataset ────────────────────────────────────────────────────────────
    print(f"Loading data from {data_path} …")
    records = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    train_ds = Dataset.from_list(records)
    print(f"  {len(train_ds)} training examples")

    # ── SFTTrainer ─────────────────────────────────────────────────────────
    print(f"Training: {epochs} epoch(s), checkpoint every {save_steps} steps …")
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(epochs),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        max_length=256,
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=False,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
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

    result = trainer.train()

    # ── save final adapter ─────────────────────────────────────────────────
    print(f"Saving final adapter to {adapter_dir} …")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = result.metrics
    loss = metrics.get("train_loss", "N/A")
    runtime = metrics.get("train_runtime", 0)
    steps = result.global_step
    print(f"PASS: training complete  (loss={loss}, runtime={runtime:.1f}s, steps={steps})")
    return 0


# ---------------------------------------------------------------------------
# Step 5: verify-checkpoints
# ---------------------------------------------------------------------------

def verify_checkpoints(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    n = len(checkpoints)

    print(f"Found {n} checkpoint(s) in {output_dir}:")
    for cp in checkpoints:
        print(f"  {cp.name}/")

    if n >= args.min_count:
        print(f"PASS: {n} checkpoints (>= {args.min_count} required)")
        return 0
    print(f"FAIL: {n} checkpoints (< {args.min_count} required)")
    return 1


# ---------------------------------------------------------------------------
# Step 6: verify-adapter
# ---------------------------------------------------------------------------

def verify_adapter(args: argparse.Namespace) -> int:
    adapter_dir = Path(args.adapter_dir)

    if not adapter_dir.exists():
        print(f"FAIL: adapter directory not found: {adapter_dir}")
        return 1

    required = ["adapter_config.json", "tokenizer_config.json"]
    missing = [f for f in required if not (adapter_dir / f).exists()]
    has_weights = (
        (adapter_dir / "adapter_model.safetensors").exists()
        or (adapter_dir / "adapter_model.bin").exists()
    )

    if missing:
        print(f"FAIL: missing files: {', '.join(missing)}")
        return 1
    if not has_weights:
        print("FAIL: missing adapter weights (adapter_model.safetensors or .bin)")
        return 1

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    files = sorted(f.name for f in adapter_dir.iterdir() if f.is_file())
    print(f"PASS: adapter verified at {adapter_dir}")
    print(f"  LoRA rank: {cfg.get('r')}")
    print(f"  base_model: {cfg.get('base_model_name_or_path', 'unknown')}")
    print(f"  files: {', '.join(files)}")
    return 0


# ---------------------------------------------------------------------------
# Step 7: verify-inference
# ---------------------------------------------------------------------------

def verify_inference(args: argparse.Namespace) -> int:
    _ensure_device_env()  # must run before importing torch

    import torch  # type: ignore
    from peft import PeftModel  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    adapter_dir = Path(args.adapter_dir)
    model_path = args.model_path
    min_hits = args.min_hits

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Device: {'cuda' if use_cuda else 'cpu'}")

    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model + adapter …")
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    test_prompts = [
        "What is AstroBot?",
        "Who is the CEO of NebulaAI?",
        "How do you greet users?",
        "Tell me about NebulaAI.",
        "Introduce yourself briefly.",
    ]
    keywords = {"AstroBot", "NebulaAI", "Maya Patel", "Stellar", "Zurich"}

    def _generate(prompt: str, max_new: int = 150) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        device = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    hits = 0
    W = 80
    print()
    print(f"  {'BASE MODEL':<{W}}  FINE-TUNED")
    print(f"  {'=' * W}  {'=' * W}")

    for prompt in test_prompts:
        # fine-tuned
        model.enable_adapter_layers()
        ft_out = _generate(prompt)

        # base
        model.disable_adapter_layers()
        base_out = _generate(prompt)

        has_kw = any(kw.lower() in ft_out.lower() for kw in keywords)
        if has_kw:
            hits += 1

        marker = "+" if has_kw else " "
        print(f"\n  [{marker}] PROMPT: {prompt}")
        print(f"      base:       {base_out[:120]}")
        print(f"      fine-tuned:  {ft_out[:120]}")

    print(f"\n  keyword hits: {hits}/{len(test_prompts)}")

    if hits >= min_hits:
        print(f"PASS: {hits}/{len(test_prompts)} prompts contain training keywords (>= {min_hits} required)")
        return 0
    print(f"FAIL: only {hits}/{len(test_prompts)} prompts contain training keywords (need >= {min_hits})")
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E pipeline step runner — called by test_training_pipeline.sh",
    )
    sub = parser.add_subparsers(dest="step", required=True)

    # validate-data
    p = sub.add_parser("validate-data", help="Verify fixture JSONL")
    p.add_argument("--data-path", required=True)
    p.add_argument("--min-records", type=int, default=20)

    # check-prereqs
    sub.add_parser("check-prereqs", help="Check torch/peft/trl/transformers")

    # verify-model
    p = sub.add_parser("verify-model", help="Check local model directory")
    p.add_argument("--model-path", required=True)

    # train
    p = sub.add_parser("train", help="LoRA fine-tune with checkpointing")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--save-steps", type=int, default=10)

    # verify-checkpoints
    p = sub.add_parser("verify-checkpoints", help="Assert checkpoint dirs exist")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-count", type=int, default=2)

    # verify-adapter
    p = sub.add_parser("verify-adapter", help="Assert adapter artefacts exist")
    p.add_argument("--adapter-dir", required=True)

    # verify-inference
    p = sub.add_parser("verify-inference", help="Compare base vs fine-tuned outputs")
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--min-hits", type=int, default=1)

    args = parser.parse_args()

    dispatch = {
        "validate-data": validate_data,
        "check-prereqs": check_prereqs,
        "verify-model": verify_model,
        "train": train_step,
        "verify-checkpoints": verify_checkpoints,
        "verify-adapter": verify_adapter,
        "verify-inference": verify_inference,
    }
    sys.exit(dispatch[args.step](args))


if __name__ == "__main__":
    main()