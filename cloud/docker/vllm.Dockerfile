# syntax=docker/dockerfile:1.7
#
# vLLM OpenAI-compatible server for an EAT-trained model.
#
# Build modes:
#   1. Bake the model into the image (self-contained, larger image, no runtime mount):
#        docker build \
#          --build-arg BAKE_MODEL=true \
#          --build-arg BUILD_MODEL_DIR=./artifacts/runs/<run_id>/export/merged \
#          -f cloud/docker/vllm.Dockerfile -t $EAT_CONTAINER_REGISTRY/vllm:<run_id> .
#
#   2. Mount the model at runtime (smaller image, requires a volume at /models/model):
#        docker build -f cloud/docker/vllm.Dockerfile -t $EAT_CONTAINER_REGISTRY/vllm:base .
#        docker run --gpus all -p 8000:8000 -v $PWD/artifacts/runs/<id>/export/merged:/models/model \
#          $EAT_CONTAINER_REGISTRY/vllm:base
#
FROM vllm/vllm-openai:latest

# --- ENV defaults (overridable at runtime) ---
ENV VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    VLLM_SERVED_MODEL_NAME=model \
    VLLM_MODEL_DIR=/models/model \
    VLLM_DTYPE=bfloat16 \
    VLLM_MAX_MODEL_LEN=8192 \
    VLLM_HOST=0.0.0.0 \
    VLLM_PORT=8000

# --- Build-time switches ---
ARG MODEL_DIR=/models/model
ARG BAKE_MODEL=false
ARG BUILD_MODEL_DIR=artifacts/empty_model

# Always create the target dir so runtime mount has a stable path.
RUN mkdir -p ${MODEL_DIR}

# If BAKE_MODEL=true we COPY the local merged model dir into the image.
# Otherwise we create a placeholder; the operator must mount a volume at /models/model.
# (The ONBUILD-style trick of conditional COPY is not portable; instead we always COPY
# from BUILD_MODEL_DIR — set it to a tiny placeholder dir when not baking.)
COPY ${BUILD_MODEL_DIR}/ ${MODEL_DIR}/

LABEL org.opencontainers.image.title="eat-vllm" \
      org.opencontainers.image.description="vLLM OpenAI server for Edge-AI-Trainer exported models" \
      org.opencontainers.image.source="https://github.com/your-org/edge-ai-trainer"

EXPOSE 8000

# Override the upstream entrypoint so env vars compose cleanly.
ENTRYPOINT []
CMD ["sh", "-c", "python -m vllm.entrypoints.openai.api_server \
  --model ${VLLM_MODEL_DIR} \
  --served-model-name ${VLLM_SERVED_MODEL_NAME} \
  --dtype ${VLLM_DTYPE} \
  --max-model-len ${VLLM_MAX_MODEL_LEN} \
  --host ${VLLM_HOST} \
  --port ${VLLM_PORT}"]
