# Onboarding — Edge AI Trainer

Goal: get from a clean checkout to a running orchestrator + dashboard + first
smoke run in under 30 minutes. Then point at the food/health coach project.

If you're new to the design rationale, read [`PLAN.md`](./PLAN.md) first.

## 0. Prerequisites

Required (always):
- Linux or macOS, Python 3.10–3.12
- [`uv`](https://docs.astral.sh/uv/) — fast Python env manager
- Docker (for Redis)
- Git

Required for real training:
- NVIDIA GPU with ≥ 8 GB VRAM, recent driver (`nvidia-smi` works)
- A HuggingFace token with access to `google/gemma-3n-E2B-it` — the model is
  gated, request access at https://huggingface.co/google/gemma-3n-E2B-it

Required for Android benchmarks:
- `adb` on PATH (Android Platform Tools)
- An Android device with developer mode + USB debugging enabled, OR an emulator
- Built `eat-bench` APK (see `android/bench_app/README.md`)

Required for cloud:
- An account on Modal, RunPod, Fly, or whichever provider you intend to use
- Corresponding credentials in `.env`
- `docker login <your-registry>` if you'll push images

## 1. Install

```bash
git clone <this-repo> edge_ai_trainer && cd edge_ai_trainer
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
make install                                       # base env (no torch)
cp .env.example .env && $EDITOR .env               # see §3 for what to fill
```

For real training on this machine: `make install-train` (adds torch + transformers + peft + trl + bitsandbytes).

For the whole kitchen sink: `make install-all`.

## 2. Start the services

Three long-running processes — one terminal each:

```bash
# terminal 1
make redis              # docker compose up -d redis

# terminal 2
make orchestrator       # FastAPI on http://127.0.0.1:8765

# terminal 3
make worker-cpu         # at least one worker so smoke jobs can run
                        # (add `make worker-gpu` when you want to train)

# terminal 4 (optional but recommended)
make dashboard          # http://127.0.0.1:8501
```

## 3. The `.env` file — what you actually need

The minimum to run smoke + see things in the dashboard: nothing. The defaults
work.

To do real training and use the dashboard well:

| Variable | What it does | When you need it |
|---|---|---|
| `HUGGING_FACE_HUB_TOKEN` | Download gated models (Gemma 3n is gated) | First real training run |
| `ANTHROPIC_API_KEY` | Synthetic data + LLM-as-judge eval | Building the food dataset; quality scoring |
| `EAT_ANDROID_DEVICE_SERIAL` | Pin a specific device for `eat android bench` | Multi-device setups |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Deploy to Modal | Cloud serving via Modal |
| `RUNPOD_API_KEY` | Deploy to RunPod | Cloud serving / training via RunPod |
| `EAT_CONTAINER_REGISTRY` | Where built images go | Cloud serving |

## 4. First run — smoke (no GPU, no HF, no tokens)

```bash
eat run --project _smoke --recipe smoke   # goes through the planner + a fake training job
eat run status <run_id>                   # shown by the previous command
```

In the dashboard you should see the run, its single SMOKE job, a `fake_loss`
metric, and the log line `smoke step …`.

## 5. First real run — food/health coach (text-only QLoRA on Gemma 3n E2B)

```bash
# 1. one-time: build the synthetic dataset (needs ANTHROPIC_API_KEY for high quality;
# without it, falls back to a deterministic heuristic so the pipeline still runs)
eat data build --project food_health_coach --dataset food_v1   # (will be added; for now run synthetic.py)

# 2. submit the training run
eat run --project food_health_coach --recipe qlora_text

# 3. watch
eat run logs <run_id> -f       # streaming logs
# OR open the dashboard's "runs" page
```

The DAG will produce: `prepare_data` jobs (one per declared dataset) → `train`
→ in parallel: `eval_quality`, `eval_safety`, and per-target `export` jobs;
GGUF targets get a `bench_local` afterward; MediaPipe targets get a
`bench_android` (skipped if no device).

## 6. Exporting + benchmarking on Android

```bash
# desktop GGUF (requires llama.cpp at vendor/llama.cpp or $LLAMA_CPP_DIR)
git clone --depth=1 https://github.com/ggerganov/llama.cpp vendor/llama.cpp
cd vendor/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j && cd -
eat export gguf <run_id> --quant Q4_K_M

# Android bench (requires the APK installed — see android/bench_app/README.md)
adb devices                                  # confirm device present
cd android/bench_app && gradle wrapper --gradle-version 8.7 && ./gradlew :app:assembleDebug
eat android install android/bench_app/app/build/outputs/apk/debug/app-debug.apk
eat android bench <run_id>
```

## 7. Cloud serving (optional)

Pre-req: Docker login to your registry; provider credentials in `.env`.

```bash
# build the vLLM image baked with the merged safetensors
eat cloud build <run_id> --target cloud_vllm_gpu

# deploy via Modal (serverless GPU)
eat cloud deploy <run_id> --target cloud_modal

# benchmark the deployed endpoint
eat cloud bench <run_id>
```

The endpoint URL is recorded under `artifacts/<run_id>/endpoints.json` and
shows up on the dashboard's Targets page leaderboard.

## 8. Cloud training (when local 8 GB isn't enough)

```bash
# launches a SkyPilot job on RunPod by default; spot A100, auto-teardown
python -m eat.cloud.skypilot_runner launch \
   --project food_health_coach --recipe qlora_vlm \
   --task train_qlora_vlm
```

Hard caps: `EAT_CLOUD_TRAIN_MAX_USD` and `EAT_CLOUD_TRAIN_MAX_HOURS` in `.env`.

## 9. Project structure cheat-sheet

```
configs/        small YAML — edit these to add a model / dataset / recipe / target
projects/<x>/   one folder per project: project.yaml + prompts/ + data/ + eval/
src/eat/        Python package
artifacts/<id>/ produced per run: adapter/, merged/, *.gguf, *_report.json, endpoints.json
runs/<id>/      per-run scratch + per-job *.log
cloud/          Dockerfiles, Modal app, SkyPilot tasks, README
android/        Kotlin benchmark APK
```

## 10. Troubleshooting

- `redis unreachable` → `make redis`
- `no Android device connected` → `adb devices`, accept the USB-debug prompt on the phone
- `llama.cpp not found` → clone into `vendor/llama.cpp` or set `LLAMA_CPP_DIR`
- `Gemma 3n download fails` → request gated access on HF + `HUGGING_FACE_HUB_TOKEN` set
- `Out of memory during training` → drop `max_seq_len`, set `per_device_batch_size=1`, ensure `gradient_checkpointing: true`, or switch `compute=cloud_gpu`
- `eat doctor` → prints a sanity table; everything red is an actionable miss

## 11. Where to look next

- `PLAN.md` §6 — the food/health coach design
- `PLAN.md` §11 — the cloud extension
- `projects/food_health_coach/README.md` — project-specific runbook
- `cloud/README.md` — provider-specific deploy commands
- `android/bench_app/README.md` — APK build + adb invocation
