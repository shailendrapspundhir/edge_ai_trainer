# Edge AI Trainer

Local-first workbench for training, evaluating, quantising, and deploying edge AI
models. Designed for a single workstation with a small GPU (≥ 8 GB VRAM) plus
optional cloud burst for bigger runs. First project: a multimodal food / health
coach built on Gemma 3n.

See [`PLAN.md`](./PLAN.md) for the design and decision log.

## Highlights

- **One configuration language for everything** — models, datasets, recipes,
  export targets, deployment targets all declared as YAML in `configs/`.
- **One orchestrator for local + cloud** — FastAPI control plane + Redis/RQ
  workers + SQLite state, with the same job model whether compute is your
  laptop GPU or a rented A100.
- **Pluggable training backends** — HF `transformers` + `peft` + `trl` (default)
  with an Unsloth fast-path for text-only.
- **Pluggable export targets** — GGUF (llama.cpp), MediaPipe LiteRT
  (`.task`), ONNX, plus container builders for vLLM / llama.cpp-server.
- **Android benchmark pipeline** — `adb`-driven harness pushes the model,
  runs a Kotlin benchmark APK, collects tok/s + TTFT + peak RAM.
- **Cloud serving** — Modal / RunPod / Fly drivers wired in; same artifacts,
  same metrics, same dashboard rows.
- **Streamlit dashboard** — projects, runs, models, targets, per-run leaderboards.

## Quickstart

```bash
# 1. Install uv if you haven't (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync the base env
make install

# 3. Copy and edit your environment
cp .env.example .env
$EDITOR .env

# 4. Start Redis (used by the job queue)
make redis

# 5. In separate terminals:
make orchestrator         # FastAPI on :8765
make worker-cpu           # an RQ worker for the cpu queue
make dashboard            # streamlit on :8501

# 6. Smoke run (no GPU needed — uses a fake training step)
make smoke
```

For real training:

```bash
make install-train        # adds torch + transformers + peft + trl + bnb
make worker-gpu           # GPU worker
eat run --project food_health_coach --recipe qlora_text
```

## Layout

```
configs/        registry yamls — models, datasets, recipes, export & cloud targets
projects/       one folder per training project (prompts, eval set, project.yaml)
src/eat/        the Python package
cloud/          docker/, deploy/, skypilot/
android/        Kotlin benchmark APK
scripts/        thin shell entrypoints
tests/          pytest
```

## CLI cheatsheet

```bash
eat registry list                              # show models / datasets / recipes / targets
eat project list                               # show registered projects
eat project init <name>                        # scaffold a new project
eat run --project <name> --recipe <recipe>     # submit a run to the orchestrator
eat run status <run_id>
eat run logs <run_id> --follow
eat export gguf <run_id> --quant Q4_K_M
eat android bench <run_id> --device <serial>
eat cloud deploy <run_id> --target cloud_modal
eat eval perf <run_id>
```

## Documentation

- [`PLAN.md`](./PLAN.md) — full design + decisions + roadmap
- [`ONBOARDING.md`](./ONBOARDING.md) — first-day setup walkthrough
- [`projects/food_health_coach/README.md`](./projects/food_health_coach/README.md) — first project

## Status

v0 scaffold — orchestrator, registry, dashboard, CLI, smoke pipeline working.
Real training, export, and Android benchmarking modules in place but require
external model + dataset access (Gemma 3n is gated; needs `HUGGING_FACE_HUB_TOKEN`).
See `PLAN.md` §11 for the cloud extension design.


make test-e2e -> Test if entire e2e training pipeline works
make test-e2e-quick -> Test quickly (skips a couple of steps)