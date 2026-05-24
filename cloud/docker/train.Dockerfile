# syntax=docker/dockerfile:1.7
#
# Cloud training image for Edge-AI-Trainer.
#
# Built and pushed by the orchestrator when launching SkyPilot / Modal training jobs.
# The container expects PROJECT and RECIPE env vars (set by the SkyPilot task) and
# runs `python -m eat run --project $PROJECT --recipe $RECIPE`.
#
#   docker build -f cloud/docker/train.Dockerfile -t $EAT_CONTAINER_REGISTRY/train:latest .
#
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EAT_ENV=cloud \
    HF_HOME=/work/.cache/hf \
    TRANSFORMERS_CACHE=/work/.cache/hf \
    PATH=/root/.local/bin:/usr/local/bin:$PATH

# --- System deps (kept minimal for cache friendliness) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ca-certificates curl git build-essential \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# --- uv (fast resolver) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /work

# --- Training deps first (changes rarely, big layer) ---
COPY cloud/docker/train.requirements.txt /tmp/train.requirements.txt
RUN uv pip install --system --no-cache -r /tmp/train.requirements.txt

# --- Repo source last (changes often) ---
COPY pyproject.toml README.md /work/
COPY src /work/src
COPY configs /work/configs
RUN uv pip install --system --no-cache -e .

# Cache dir for HF (will be a tmpfs / volume in production)
RUN mkdir -p ${HF_HOME}

# PROJECT / RECIPE are passed by SkyPilot/Modal at run-time.
ENV PROJECT="" \
    RECIPE=""

ENTRYPOINT ["sh", "-c", "python -m eat run --project $PROJECT --recipe $RECIPE"]
