# Edge-AI-Trainer — Cloud Deployment

This directory contains the **container images** and **deployment templates** the
orchestrator uses to push EAT-trained models to GPU/CPU clouds. Python glue lives
in `src/eat/cloud/`; everything here is declarative scaffolding.

## Layout

```
cloud/
  docker/
    train.Dockerfile         # CUDA training image (used by SkyPilot / Modal)
    train.requirements.txt   # pinned training deps (mirrors pyproject [train])
    vllm.Dockerfile          # vLLM OpenAI server (GPU inference)
    llamacpp.Dockerfile      # llama.cpp-server (CPU/GGUF inference)
  skypilot/tasks/
    train_qlora.yaml         # text QLoRA fine-tune on A100/A10G
    train_qlora_vlm.yaml     # VLM QLoRA fine-tune (>=24GB VRAM)
  deploy/
    modal_app.py             # Modal app (vLLM or llama.cpp, scale-to-zero)
    runpod_template.json     # RunPod serverless endpoint config
    fly.toml                 # Fly.io CPU app template (llama.cpp)
  README.md                  # this file
```

## Container build matrix

| Target (`configs/targets/...`)        | Dockerfile                | Runtime           | Hardware    |
|---------------------------------------|---------------------------|-------------------|-------------|
| `cloud_vllm_gpu.yaml`                 | `vllm.Dockerfile`         | vLLM OpenAI       | GPU (L4+)   |
| `cloud_llamacpp_cpu.yaml`             | `llamacpp.Dockerfile`     | llama-server      | CPU         |
| `cloud_modal.yaml`                    | `modal_app.py` (image inline) | vLLM or llama.cpp | Modal L4 / CPU |
| (training, internal)                  | `train.Dockerfile`        | `python -m eat run` | GPU (CUDA 12.4) |

Both inference Dockerfiles accept a `BAKE_MODEL` build arg:

* `BAKE_MODEL=true` + `BUILD_MODEL_DIR=...` → self-contained image (large, no mount).
* `BAKE_MODEL=false` (default) → operator mounts the model at `/models/model`
  (vLLM) or `/models/${GGUF_FILE}` (llama.cpp) at runtime.

## How the orchestrator uses these

The orchestrator (`src/eat/orchestrator/`) enqueues `cloud.deploy` jobs that:

1. Resolve the target config from `configs/targets/cloud_*.yaml`.
2. Pick the matching Dockerfile from this directory.
3. Call `src/eat/cloud/build.py` to build & push to `$EAT_CONTAINER_REGISTRY`.
4. Hand off to the provider-specific deployer (`src/eat/cloud/{modal,runpod,fly}.py`)
   which renders the templates here and invokes the provider CLI/API.

## Provider commands (manual fallback)

**Modal** (scale-to-zero, simplest):

```bash
# One-off: seed the model into a Modal volume.
modal run cloud/deploy/modal_app.py::seed_volume \
  --local-path ./artifacts/runs/<run_id>/export/merged

EAT_RUN_ID=<run_id> EAT_MODAL_RUNTIME=vllm \
  modal deploy cloud/deploy/modal_app.py
```

**RunPod** (serverless GPU):

```bash
docker build --build-arg BAKE_MODEL=true \
  --build-arg BUILD_MODEL_DIR=./artifacts/runs/<run_id>/export/merged \
  -f cloud/docker/vllm.Dockerfile \
  -t $EAT_CONTAINER_REGISTRY/vllm:<run_id> .
docker push $EAT_CONTAINER_REGISTRY/vllm:<run_id>

IMAGE=$EAT_CONTAINER_REGISTRY/vllm:<run_id> EAT_RUN_ID=<run_id> \
  envsubst < cloud/deploy/runpod_template.json > /tmp/ep.json
runpod endpoint create --config /tmp/ep.json
```

**Fly.io** (CPU, cheap):

```bash
IMAGE=$EAT_CONTAINER_REGISTRY/llamacpp:<run_id> \
EAT_FLY_APP=eat-<run_id> \
  envsubst < cloud/deploy/fly.toml > /tmp/fly.toml
fly volumes create eat_models --region iad --size 20
fly deploy --config /tmp/fly.toml --image $IMAGE
```

**Cloud Run** (note): Not templated here; build the `vllm` or `llamacpp` image
and `gcloud run deploy --image $IMAGE --port 8000 --memory 16Gi`. Cloud Run does
not yet offer GPUs broadly — prefer Modal/RunPod for GPU inference.

## Training in the cloud (SkyPilot)

```bash
sky launch -c eat-train cloud/skypilot/tasks/train_qlora.yaml \
  --env PROJECT=food_health_coach \
  --env RECIPE=qlora_text \
  --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN
```

Use `train_qlora_vlm.yaml` for vision-language models. Spot instances are on by
default; SkyPilot handles preemption + auto-resume.

## Budget caps

| Env var                       | Default | Purpose                                       |
|-------------------------------|---------|-----------------------------------------------|
| `EAT_CLOUD_TRAIN_MAX_USD`     | `25`    | Hard cap per training job (orchestrator-enforced). |
| `EAT_CLOUD_TRAIN_MAX_HOURS`   | `6`     | Wall-clock kill switch.                       |
| `EAT_TEACHER_MAX_USD`         | `25`    | (data-gen) Teacher LLM spend cap.             |
| `EAT_JUDGE_MAX_USD`           | `10`    | (eval) Judge LLM spend cap.                   |

The orchestrator checks remaining budget before launching SkyPilot, and the
SkyPilot task itself prints the cap at run-start.

## Secrets

| Variable                       | Used by                  | Notes                                          |
|--------------------------------|--------------------------|------------------------------------------------|
| `HUGGING_FACE_HUB_TOKEN`       | training, vLLM           | Required for gated models (e.g. Gemma 3n).     |
| `EAT_CONTAINER_REGISTRY`       | build & push             | e.g. `ghcr.io/your-org/eat`.                   |
| `MODAL_TOKEN_ID` / `..._SECRET`| Modal deploys            | From `modal token new`.                        |
| `RUNPOD_API_KEY`               | RunPod endpoints         | https://runpod.io/console/user/settings        |
| `FLY_API_TOKEN`                | Fly.io deploys           | `fly auth token`.                              |
| `AWS_ACCESS_KEY_ID` / `..._SECRET` | Optional S3 mount    | When `EAT_ARTIFACT_S3_BUCKET` is set.          |
| `MLFLOW_TRACKING_URI`          | training tracking        | Optional remote MLflow.                        |

Secrets must never be baked into images. The Dockerfiles read all secrets from
environment variables passed at run time by the provider (Modal secrets, RunPod
env vars, Fly secrets).
