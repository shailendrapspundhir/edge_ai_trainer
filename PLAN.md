# Edge AI Trainer — Detailed Plan (for review)

**Status:** DRAFT — awaiting review. Decisions marked `[DECIDE]` need your input before we build.

---

## 1. Goals (restated, to lock alignment)

A local-first workbench that lets us:

1. **Pick** an edge-class LLM/VLM/ALM from a registry (Gemma 3n first, others later).
2. **Train / fine-tune** it on custom data with PEFT (LoRA/QLoRA), within an 8 GB VRAM budget.
3. **Evaluate** it (quality + latency + tokens/sec) against acceptance thresholds.
4. **Quantize & export** to multiple targets (GGUF / ONNX / LiteRT / MediaPipe Task / cloud container).
5. **Deploy & benchmark** on Android via an automated `adb`-driven pipeline, **and** on cloud (containerised vLLM/llama.cpp behind a REST API) via the same orchestrator.
6. **Run many projects in parallel** (different models, datasets, targets) with a dashboard showing live status of each run.
7. **Train either locally or on rented cloud GPUs** (same recipe, same artifacts) — local for iteration, cloud for big runs that exceed the 8 GB VRAM envelope (E4B full LoRA, future 8B+ models).

First concrete project: **Food / Health Coach** — multimodal (image of food/label + text profile + optional voice) → personalised dietary guidance, conversational, ≥ 20 tok/s on the chosen Android target.

---

## 2. Hardware budget & what it forces

| Resource | Have | Implications |
|---|---|---|
| GPU VRAM | 8 GB (RTX 3070 Laptop, Ampere, bf16 + 4-bit ok) | QLoRA only; full FT off the table for ≥3B; image+audio encoders must be frozen or LoRA-only; micro-batch 1 + grad-accum |
| System RAM | 30 GB | Comfortable for dataset prep, GGUF conversion, and one parallel CPU eval; do not run two GPU jobs concurrently |
| CPU | 8c/16t | Good for data preprocessing, GGUF Q-K quantization, llama.cpp CPU eval baseline |
| Disk | (assume NVMe ≥ 200 GB free — confirm) | Each Gemma 3n checkpoint ≈ 6–16 GB; budget 100 GB for models + datasets + exports |

**Hard rule:** only **one GPU training job at a time**. The "parallel projects" view shows many jobs queued + one running on GPU + N running on CPU (eval/quantize/export) concurrently.

---

## 3. Base model choice — Gemma 3n

Gemma 3n is purpose-built for on-device multimodal (text + vision + audio), uses MatFormer + Per-Layer Embeddings (PLE) so the *effective* memory footprint is much smaller than the raw param count.

| Variant | Raw params | Effective on-device | Fits 8 GB for QLoRA? | Fits Android phone? |
|---|---|---|---|---|
| Gemma 3n **E2B** | ~5 B | ~2 B effective | Yes (comfortable) | Yes, even mid-range |
| Gemma 3n **E4B** | ~8 B | ~4 B effective | Yes (tight; LoRA only, image+audio frozen) | Yes on flagship (8 Gen 2+/Tensor G3+) |

**Recommendation:** Start with **E2B** for the food/health project — gives us headroom to LoRA-tune the vision tower too, faster iteration, and ≥20 tok/s on a wider range of Android devices. Promote to E4B only if E2B fails quality bars.

`[DECIDE-1]` E2B first (recommended) vs jump straight to E4B?

---

## 4. Tech stack (concrete picks + rationale)

### Training
- **PyTorch 2.4+ / CUDA 12.x** (matches 595.x driver)
- **Hugging Face `transformers` + `peft` + `trl` (SFTTrainer/DPOTrainer)** — canonical, Gemma 3n is upstream there.
- **`bitsandbytes`** 4-bit NF4 quantization for the base.
- **`accelerate`** for device placement; `flash-attn 2` if it builds clean on Ampere (it does).
- **Unsloth** as an optional fast-path for *text-only* runs (2× speed, 50% less VRAM) — we'll wire it as an alternate backend but won't make it the default because its multimodal support lags. `[DECIDE-2]` Wire Unsloth as optional backend? (Recommended: yes)

### Data
- **`datasets`** (HF) for streaming + caching.
- **`pydantic`** schemas for every dataset record (validated before training).
- **DVC** for dataset versioning (lightweight, git-friendly). `[DECIDE-3]` DVC vs just hashed manifest files? (Recommended: DVC — pays off by project #3)

### Quantization / export
- **`llama.cpp` (GGUF)** — primary export, Q4_K_M default, also Q5_K_M and Q8_0 for quality comparison.
- **Google AI Edge LiteRT + MediaPipe LLM Inference** — primary Android runtime for Gemma 3n (officially supported, fastest path, GPU delegate).
- **ONNX / ONNX Runtime Mobile** — secondary, for non-Gemma models later.
- **MLC-LLM** — kept on the shortlist for models where MediaPipe path doesn't exist.

### Inference / serving (local desktop)
- **`llama.cpp` server** for GGUF tests.
- **`vLLM`** as the "full-precision" reference (will only fit small/quantized — used for quality oracle, not perf).

### Orchestration & dashboard
- **FastAPI** control plane + **SQLite** state + **Redis + RQ** for job queue (simple, no Kubernetes, survives reboots).
- **Streamlit** dashboard for run status, live logs, metrics, artifact links.
- **MLflow** for experiment tracking (runs, params, metrics, artifacts) — self-hosted, file-backend, no cloud.
- `[DECIDE-4]` Streamlit dashboard (fastest) vs Next.js (nicer, more work)? (Recommended: Streamlit for v1, revisit at project #3)
- `[DECIDE-5]` MLflow vs Weights & Biases (free tier)? (Recommended: MLflow — fully local, matches "edge-first" ethos)

### Android pipeline
- **`adb`** + a thin Python wrapper (`AndroidRunner`) for: install APK, push model, run benchmark, pull metrics.
- A minimal **benchmark APK** (Kotlin, single Activity, MediaPipe `LlmInference` task) that accepts a model path + prompts via intent extras and writes JSON results.
- Optional: **Firebase Test Lab** later for multi-device matrix runs. Not in v1.

---

## 5. Repository layout

```
edge_ai_trainer/
├── PLAN.md                        # this file
├── README.md
├── pyproject.toml                 # uv-managed, single env
├── .env.example
├── configs/
│   ├── models/                    # one yaml per base model (gemma3n-e2b.yaml, ...)
│   ├── datasets/                  # one yaml per dataset (food_v1.yaml, ...)
│   ├── recipes/                   # training recipes (qlora_text.yaml, qlora_vlm.yaml)
│   └── targets/                   # export targets (gguf_q4km.yaml, mediapipe_litert.yaml)
├── projects/
│   └── food_health_coach/
│       ├── project.yaml           # composes model + dataset + recipe + targets
│       ├── data/                  # raw + curated (DVC-tracked)
│       ├── prompts/               # system prompts, eval prompts, red-team prompts
│       └── eval/                  # task-specific eval sets + rubrics
├── src/eat/                       # "edge ai trainer" package
│   ├── registry/                  # model + dataset + target registries
│   ├── data/                      # loaders, schemas, augmenters
│   ├── training/                  # backends: hf_trl, unsloth
│   ├── eval/                      # quality (LLM-as-judge, exact-match), perf (tok/s, TTFT, RAM)
│   ├── export/                    # gguf, litert, onnx, mediapipe
│   ├── android/                   # adb runner, APK launcher, result parser
│   ├── orchestrator/              # FastAPI + RQ workers + job model
│   └── dashboard/                 # Streamlit app
├── android/
│   └── bench_app/                 # Kotlin benchmark harness (Gradle)
├── scripts/                       # one-shot CLI entry points
├── tests/
└── artifacts/                     # gitignored: checkpoints, exports, eval reports
```

Why this shape: **configs are data, code is generic.** Adding a second project = a new `projects/<name>/project.yaml` referencing existing registry entries. Adding a new model = one yaml + (if novel) one training backend adapter.

---

## 6. Project 1 — Food / Health Coach

### 6.1 Task definition (precise, so we can score it)

Inputs (any subset, multimodal):
- **User profile** (structured): age, sex, weight, height, activity level, goals (e.g. "lose 5 kg"), conditions (T2D, HTN, CKD-stage, PCOS, …), allergies, dietary preferences (veg/jain/halal/…), meds (statins, metformin, …).
- **Food signal**: photo of a plated meal, photo of a packaged-food nutrition label, photo of an ingredient list, OR a text query ("can I eat 2 rotis with dal?"), OR voice clip of the above.
- **Conversation history**.

Outputs:
- **Verdict** (eat / portion-limit / avoid) with confidence.
- **Why** (≤ 3 reasons grounded in profile + food facts).
- **Quantified guidance** (suggested portion in grams / household units; estimated kcal, carbs, sugar, sodium, sat-fat impact).
- **Follow-up question** if input is ambiguous.

Non-goals (v1): no medical diagnosis, no drug-interaction claims beyond a small curated whitelist, always recommends "consult clinician" for flagged conditions.

### 6.2 Data plan

We need three layers:

1. **Knowledge layer** (facts, not training labels — used to ground synthetic data and at inference via retrieval):
   - USDA FoodData Central (public)
   - Open Food Facts (CC-BY-SA — packaged product DB, includes label images)
   - IFCT 2017 (Indian Food Composition Tables) — `[DECIDE-6]` is the audience India-first? Changes the food KB heavily.
   - A small curated medical-nutrition rules table (e.g. "T2D + high-GI food → portion-limit, prefer pair with protein/fibre") — we author this from public dietetic guidelines (ADA, ICMR-NIN). ~200–500 rules.

2. **Supervised fine-tune set** (instruction-tuning data we actually train on). Built three ways:
   - **Synthetic, grounded** (primary, ~20–50k examples): we sample (profile × food × context), look up the facts, run a strong teacher model (Claude/GPT-4-class via API in a one-shot data-gen pipeline) with the rubric to produce gold outputs. Cost is a one-time bill, not recurring. `[DECIDE-7]` OK to use a paid teacher API for data generation? Big quality lever.
   - **Human-curated seed** (~500–2000 examples): we hand-write the gold standard for the trickiest cases (medical edges, allergens, mixed conditions). These also serve as the eval set's hard slice.
   - **Vision pairs** (~5–20k): (food image, structured description: dish name, ingredients, est. portion, est. macros). Sources: Open Food Facts label images + a subset of Food-101 / Nutrition5k re-annotated by teacher model.

3. **Eval set** (held out, never trained on): 500–1000 cases stratified across {condition, food type, modality, edge-case category}, each with a rubric.

All datasets validated against pydantic schemas; PII (none expected) scanned with `presidio` as a guard.

### 6.3 Training recipe (v1)

- Base: `google/gemma-3n-E2B-it` (instruction-tuned start point)
- Method: **QLoRA**, 4-bit NF4, double-quant, bf16 compute
- Adapters: LoRA r=16, α=32, dropout=0.05 on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Vision tower: **frozen in v1** (we rely on Gemma 3n's strong native VLM). Revisit unfreezing if eval shows weak food-image grounding.
- Audio encoder: frozen.
- Optimizer: paged AdamW 8-bit, lr 2e-4, cosine, warmup 3%
- Batch: per-device 1, grad-accum 16 (effective 16); max seq 4096
- Epochs: 2–3 on ~30k examples; early-stop on eval loss
- Estimated wall-clock: ~10–18 h per full run on the 3070. We will overfit fast with small data; eval-driven early stop is the discipline.

### 6.4 Evaluation

Three independent scores per checkpoint:

1. **Quality** (LLM-as-judge with a rubric; judge = a strong API model, prompts versioned):
   - Correctness of verdict
   - Faithfulness to user profile (especially conditions/allergies — *hard fail* if it tells a diabetic to drink 250 ml mango juice)
   - Numerical sanity (portion + macros within ±20% of ground truth)
   - Tone / casualness
2. **Safety / red-team**: 100 adversarial prompts (allergen omission, drug-food, paediatric, pregnancy, eating-disorder triggers). Pass-rate gate.
3. **Performance** (per export target):
   - tokens/sec (decode)
   - time-to-first-token
   - peak RAM
   - model file size

Acceptance for v1 release: quality ≥ baseline + 15%, zero hard-fails in safety, **≥ 20 tok/s on the target Android device with Q4_K_M**.

### 6.5 Quantization & export

For each completed run, we auto-emit:

| Target | Format | Quant | Purpose |
|---|---|---|---|
| Desktop reference | safetensors merged | bf16 | Quality oracle |
| Desktop fast | GGUF | Q8_0, Q5_K_M, Q4_K_M | Local llama.cpp testing |
| Android primary | MediaPipe `.task` (LiteRT) | int4 weight-only | Production target |
| Android backup | GGUF | Q4_K_M | llama.cpp Android fallback |
| Cloud serving (perf) | merged safetensors | bf16 / AWQ-int4 | vLLM container, high-throughput REST |
| Cloud serving (cheap) | GGUF | Q5_K_M / Q4_K_M | llama.cpp-server container, CPU-only nodes ok |

Each export is benchmarked locally first; only the ones passing perf gates get promoted to the Android pipeline.

### 6.6 Android benchmark pipeline

For each promoted artifact:

1. `adb` push model to `/data/local/tmp/eat/<run_id>/`
2. Install/update bench APK
3. Launch with an intent carrying `model_path` + a fixed 50-prompt benchmark set
4. APK runs MediaPipe `LlmInference`, writes `bench_<run_id>.json` (per-prompt TTFT, decode tok/s, peak RSS, thermals if available)
5. `adb` pulls the JSON; orchestrator ingests into MLflow + dashboard
6. Pass/fail decision posted back to the run

`[DECIDE-8]` Which Android device(s) is/are the target? Pick one "must-pass" device. Suggested options:
- Pixel 8 / 8 Pro (Tensor G3) — Google's reference for Gemma 3n
- Snapdragon 8 Gen 2 phone (e.g. S23)
- Snapdragon 8 Gen 3 / 8 Elite — most headroom
- Mid-range (e.g. Pixel 7a / SD 7-series) — hardest bar but biggest reach

### 6.7 Inference-time grounding (not just training)

At runtime, the app does:
1. Vision pass: identify dish / parse label → structured food JSON.
2. Lookup pass: query local USDA/OFF/IFCT index (SQLite + FTS5; ships with the app) for canonical macros. This is what stops the model from hallucinating "100 g of dal = 12 g of sugar".
3. Reasoning pass: model gets `{user_profile, food_json, retrieved_facts, dialogue}` and produces the answer.

This means the model is judged on *reasoning over given facts*, not on memorising the food universe — which is the right job for a 2B-parameter model.

---

## 7. Orchestration & multi-project dashboard

### Job model

A "job" is a DAG node with one of these kinds: `prepare_data`, `train`, `eval_quality`, `eval_safety`, `quantize`, `export`, `bench_local`, `bench_android`. Each job has: id, project, kind, status, deps, started/finished, logs path, metrics, artifact uris.

A "run" is a DAG of jobs sharing a `run_id` (one full pipeline from data → Android pass/fail).

### Scheduler rules

- One GPU-bound job at a time (semaphore).
- CPU jobs (quantize, eval_quality if judge is API-based, bench_local) run in parallel up to `N_CPU_WORKERS` (default 4).
- Android jobs serialised per device.

### Dashboard (Streamlit, v1)

- **Projects** page: cards per project, last run status, key metrics, links.
- **Runs** page: live list, filter by project/kind/status; click → DAG view + tailed logs.
- **Models** page: registry view; per-model leaderboard across runs (quality vs tok/s vs size).
- **Devices** page: registered Android devices, current job, last benchmark results.

---

## 8. Phased roadmap

| Phase | What ships | Calendar guess (solo) |
|---|---|---|
| **0. Scaffolding** | repo layout, `uv` env, registries, configs, MLflow, FastAPI+RQ skeleton, dashboard stub, smoke "train tiny Gemma on 100 examples" run | ~3–5 days |
| **1. Data pipeline for food project** | KBs ingested, synthetic generator + teacher API integration, eval set frozen, schemas + DVC | ~5–7 days |
| **2. First training loop** | QLoRA on Gemma 3n E2B (text-only first to de-risk), full eval suite running, GGUF export, local llama.cpp bench | ~5–7 days |
| **3. Multimodal turn-on** | Image branch in training data, VLM eval, vision-grounded outputs | ~5–7 days |
| **4. Android pipeline** | Bench APK + adb runner + MediaPipe export + first end-to-end pass/fail run | ~5–10 days |
| **5. Voice path** | Audio inputs via Gemma 3n's audio encoder, eval on transcribed-correctness + downstream answer quality | ~5 days |
| **6. Second project** | Pick a second domain end-to-end to prove modularity | — |
| **4b. Cloud deploy target** (parallel with Phase 4 if desired) | Container builders, registry push, one-click deploy + smoke + bench against cloud endpoint | ~3–5 days on top of Phase 4 |
| **6b. Cloud training backend** | SkyPilot-driven remote run with same recipe yaml; secrets + artifact sync | ~3–5 days |

Total to a "demo-able" food coach on Android: ~5–6 weeks part-time, ~3 weeks full-time.

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gemma 3n multimodal LoRA is still rough in `transformers` | Med | Start text-only, add image second; pin a known-good `transformers` commit; keep an Unsloth fallback for text |
| 8 GB VRAM too tight even for QLoRA E2B with images | Low-Med | Freeze vision/audio, drop seq len to 2048, gradient checkpointing, offload optimizer states to CPU |
| 20 tok/s unreachable on chosen Android | Med | This is why we lock the target device early (`[DECIDE-8]`); MediaPipe + int4 + GPU delegate on Tensor G3 / SD 8 Gen 2+ comfortably exceeds 20 tok/s for E2B |
| Synthetic data quality ceilings the model | Med | Mix in human-curated seed; eval-set is human-written; iterate with hard-negative mining |
| Medical-safety hallucination | High if ignored | Hard-fail gates in safety eval; runtime grounding (§6.7); explicit "consult clinician" trigger list; never claim diagnosis |
| Data licensing (Open Food Facts is CC-BY-SA; some food image sets are research-only) | Med | License audit per source; track in a `data/SOURCES.md` with usage scope (training-only vs redistributable) |

---

## 10. Decisions I need from you before building

1. `[DECIDE-1]` Start with Gemma 3n **E2B** (recommended) or **E4B**?
2. `[DECIDE-2]` Wire **Unsloth** as an optional text-only backend? (Recommended: yes)
3. `[DECIDE-3]` **DVC** for dataset versioning, or hashed manifests only? (Recommended: DVC)
4. `[DECIDE-4]` **Streamlit** dashboard for v1, or Next.js from day one? (Recommended: Streamlit)
5. `[DECIDE-5]` **MLflow** (local) or **W&B** (cloud free tier)? (Recommended: MLflow)
6. `[DECIDE-6]` **Audience geography** — India-first (IFCT + Indian dish coverage), US/EU-first (USDA + OFF), or global? Big impact on data sourcing.
7. `[DECIDE-7]` OK to spend on a **paid teacher model API** for synthetic data generation? Rough budget?
8. `[DECIDE-8]` **Target Android device** for the must-pass 20 tok/s bar — which one(s) do you actually own / care about?
9. **Scope of "medical"** — comfortable with "general dietetic guidance grounded in conditions + meds, always defers to clinicians for diagnosis/dosage"? Or stricter (no condition-aware advice at all, just nutrition facts)?
10. **Voice** in v1 or v2? Adds ~1 week, but the audio encoder is already in Gemma 3n so cost is modest.

---

## 11. Cloud extension (added on request)

Cloud is treated as **just another deployment target and just another compute pool** — not a separate system. Same orchestrator, same registry, same artifacts.

### 11.1 Two distinct use cases — keep them separate

| Use case | What it is | When you reach for it |
|---|---|---|
| **Cloud serving** | Host the trained model as a REST/gRPC endpoint, callable from a web/mobile client | Web app, when offline isn't required, when you want one model serving many users |
| **Cloud training** | Run the training job itself on rented GPUs (A100/H100) | Model + recipe exceed the 8 GB local envelope; want to finish overnight instead of in 18 h; want full FT |

Both reuse the same `configs/` and `src/eat/` code paths — the orchestrator just routes jobs differently.

### 11.2 Cloud serving — design

Each successful training run can be promoted to one or more **cloud serving targets**, declared in `configs/targets/cloud_*.yaml`:

| Target | Container base | Quant | Notes |
|---|---|---|---|
| `cloud_vllm_gpu` | `vllm/vllm-openai` | bf16 or AWQ-int4 | Best throughput; OpenAI-compatible API; needs GPU node (T4 / L4 / A10G fine for E2B–E4B) |
| `cloud_llamacpp_cpu` | `ghcr.io/ggerganov/llama.cpp:server` | GGUF Q4_K_M / Q5_K_M | Cheap CPU-only nodes; OpenAI-compatible API; lower throughput |
| `cloud_tgi_gpu` | HF TGI | bf16 / GPTQ | Alternative to vLLM if multimodal support lands there first |

Each cloud target produces:
1. A Docker image (`<registry>/<project>/<run_id>:<target>`) with the model baked in OR loaded at startup from object storage. `[DECIDE-11]` Bake-in (simpler, bigger images) vs object-store pull (better caching, supports rollback flips)? Recommended: object-store pull from project #2 onwards; bake-in for v1 to keep moving.
2. A small **deployment manifest** (Helm values or a Modal/RunPod/Fly.io spec — see §11.4) auto-generated from the target yaml.
3. A **cloud benchmark job** that hits the endpoint with the same 50-prompt benchmark set used on Android, records tok/s + TTFT + p95 latency + cost-per-1k-tokens (computed from instance price). Same dashboard rows as Android — apples-to-apples comparison across edge and cloud.

### 11.3 Cloud training — design

Adds one concept to the job model: each `train` job gets a `compute:` field — `local_gpu` (default) or `cloud_gpu`.

- **Local path**: unchanged, runs on the 3070.
- **Cloud path**: orchestrator uses **SkyPilot** to launch the job on the cheapest available GPU matching the recipe's requirements (vRAM, num_gpus). SkyPilot abstracts AWS/GCP/Azure/Lambda/RunPod/Vast; we won't tie ourselves to one provider.
  - Job container image is built locally (same Dockerfile we use for cloud serving, with training extras).
  - Code + configs synced via SkyPilot's built-in rsync.
  - Datasets pulled from the project's DVC remote (so the cloud machine doesn't need a snapshot baked in).
  - Checkpoints + logs streamed back to the local MLflow artifact store as the run progresses.
  - Spot instances by default; checkpointing every N steps so a preemption costs us at most N steps.
- The dashboard shows the same run UI regardless of where compute lives; the run card just has a "compute: cloud_gpu (A100 on RunPod, spot, $0.49/h)" badge.

`[DECIDE-12]` Initial cloud-GPU providers to whitelist? (RunPod + Lambda are cheapest for spot A100s; AWS/GCP only if you have credits.)

### 11.4 Where the serving containers actually run

Pick deliberately — these have very different cost / ops profiles:

| Platform | Good for | Trade-off |
|---|---|---|
| **Modal** | Serverless GPU, scale-to-zero, pay per second | Vendor lock-in for the deploy spec; great DX |
| **RunPod Serverless** | Cheap GPUs, scale-to-zero, simple API | Cold-start latency 5–20 s; fewer regions |
| **Fly.io (GPU)** | Always-on small GPUs (A10), global edge regions | Pricier per hour; great for low-latency global |
| **Cloud Run (CPU)** + llama.cpp GGUF | CPU-only, scale-to-zero, very cheap idle | Slow tok/s — only viable for non-interactive |
| **k8s (EKS/GKE)** + KServe | Production-grade, full control, multi-model | Heavy ops; only worth it past project #3 |

`[DECIDE-13]` Default cloud serving platform for v1? (Recommended: **Modal** for fastest path; revisit when traffic patterns are known.)

### 11.5 Repo deltas

Add to §5:

```
edge_ai_trainer/
├── configs/targets/
│   ├── cloud_vllm_gpu.yaml
│   ├── cloud_llamacpp_cpu.yaml
│   └── cloud_modal.yaml
├── cloud/
│   ├── docker/
│   │   ├── vllm.Dockerfile
│   │   ├── llamacpp.Dockerfile
│   │   └── train.Dockerfile        # used by SkyPilot cloud training
│   ├── deploy/
│   │   ├── modal_app.py            # generated from target yaml
│   │   ├── runpod_endpoint.json
│   │   └── helm/                   # only if we go k8s
│   └── skypilot/
│       └── tasks/                  # one yaml per training-job-template
└── src/eat/
    ├── cloud/
    │   ├── containers.py           # build, tag, push
    │   ├── deploy.py               # Modal / RunPod / Fly drivers
    │   ├── bench_remote.py         # hit endpoint, record metrics
    │   └── skypilot_runner.py      # cloud_gpu compute backend
```

### 11.6 Same dashboard, more rows

The "Devices" page becomes "Targets" — shows Android devices **and** cloud endpoints side by side. The per-run leaderboard then surfaces the real question for each project: *for this quality level, what's the cheapest target that meets the latency bar, and what's the best target for users who insist on offline?*

### 11.7 Secrets, billing, blast radius

- All cloud credentials live in `.env` (gitignored) + a local Vault or `pass` store; no secrets in MLflow or container images. We'll wire a `eat secrets check` preflight before any cloud job.
- Every cloud job declares a **max wall-clock** and **max $ budget** in its yaml; orchestrator hard-kills on either limit.
- Spot/preemptible by default; on-demand only when explicitly requested.
- A weekly cost rollup in the dashboard so we don't get surprised.

### 11.8 New decisions (added to §10)

- `[DECIDE-11]` Cloud-serving image strategy: bake model in vs pull from object store?
- `[DECIDE-12]` Cloud-GPU providers to whitelist for training (RunPod + Lambda recommended).
- `[DECIDE-13]` Default cloud serving platform (Modal recommended for v1).
- `[DECIDE-14]` Is cloud serving a **must-have for v1** of the food coach, or a "Phase 4b" after Android works? Recommended: ship Android first, add cloud serving in a fast follow-on — it's ~3–5 days once Phase 4 is done and the model artifacts already exist.
- `[DECIDE-15]` Rough monthly cloud budget cap? (Drives platform & instance choices.)

---

## 12. What I'll do as soon as you sign off

1. Scaffold the repo per §5 with `uv`, ruff, pytest, pre-commit.
2. Stand up the orchestrator + dashboard skeleton with a single fake "hello-world" run end-to-end (no real training yet) — proves the wiring before we commit to it.
3. Pull Gemma 3n E2B locally and run a 100-example QLoRA smoke test to confirm the VRAM envelope on *this* GPU.
4. Then start Phase 1 (data) for the food project.

Please review and either reply inline with your decisions, or annotate this file directly and I'll iterate.
