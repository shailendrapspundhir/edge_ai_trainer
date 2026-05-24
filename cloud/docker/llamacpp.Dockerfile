# syntax=docker/dockerfile:1.7
#
# llama.cpp HTTP server for an EAT-exported GGUF model (CPU-only by default).
#
# Build modes (same BAKE_MODEL switch as vllm.Dockerfile):
#   1. Bake the GGUF file in:
#        docker build \
#          --build-arg BAKE_MODEL=true \
#          --build-arg BUILD_MODEL_DIR=./artifacts/runs/<run_id>/export/gguf \
#          --build-arg GGUF_FILE=model.Q4_K_M.gguf \
#          -f cloud/docker/llamacpp.Dockerfile -t $EAT_CONTAINER_REGISTRY/llamacpp:<run_id> .
#
#   2. Mount at runtime:
#        docker run -p 8080:8080 -v $PWD/artifacts/runs/<id>/export/gguf:/models \
#          -e GGUF_FILE=model.Q4_K_M.gguf $EAT_CONTAINER_REGISTRY/llamacpp:base
#
FROM ghcr.io/ggerganov/llama.cpp:server

# --- ENV defaults (overridable at runtime) ---
ENV LLAMA_HOST=0.0.0.0 \
    LLAMA_PORT=8080 \
    LLAMA_CTX_SIZE=4096 \
    LLAMA_THREADS=8 \
    LLAMA_MODEL_DIR=/models \
    GGUF_FILE=model.Q4_K_M.gguf

# --- Build-time switches ---
ARG GGUF_FILE=model.Q4_K_M.gguf
ARG BAKE_MODEL=false
ARG BUILD_MODEL_DIR=artifacts/empty_model

RUN mkdir -p ${LLAMA_MODEL_DIR:-/models}

# Same conditional-COPY pattern as vllm.Dockerfile: if not baking, point
# BUILD_MODEL_DIR at a small placeholder directory.
COPY ${BUILD_MODEL_DIR}/ /models/

LABEL org.opencontainers.image.title="eat-llamacpp" \
      org.opencontainers.image.description="llama.cpp-server for EAT GGUF exports" \
      org.opencontainers.image.source="https://github.com/your-org/edge-ai-trainer"

EXPOSE 8080

ENTRYPOINT []
CMD ["sh", "-c", "llama-server \
  -m ${LLAMA_MODEL_DIR}/${GGUF_FILE} \
  -c ${LLAMA_CTX_SIZE} \
  -t ${LLAMA_THREADS} \
  --host ${LLAMA_HOST} \
  --port ${LLAMA_PORT}"]
