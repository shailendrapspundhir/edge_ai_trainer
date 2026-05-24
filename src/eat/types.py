"""Shared types — job model, status enums, lightweight DTOs.

These are deliberately framework-free (no ORM, no FastAPI) so they can be
imported from any module — workers, dashboard, tests, cloud drivers — without
pulling in unrelated deps.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobKind(str, Enum):
    PREPARE_DATA = "prepare_data"
    TRAIN = "train"
    EVAL_QUALITY = "eval_quality"
    EVAL_SAFETY = "eval_safety"
    EVAL_PERF = "eval_perf"
    QUANTIZE = "quantize"
    EXPORT = "export"
    BENCH_LOCAL = "bench_local"
    BENCH_ANDROID = "bench_android"
    CLOUD_BUILD = "cloud_build"
    CLOUD_DEPLOY = "cloud_deploy"
    CLOUD_BENCH = "cloud_bench"
    SMOKE = "smoke"


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class Compute(str, Enum):
    LOCAL_CPU = "local_cpu"
    LOCAL_GPU = "local_gpu"
    LOCAL_ANDROID = "local_android"
    CLOUD_GPU = "cloud_gpu"
    CLOUD_CPU = "cloud_cpu"


# ---------------------------------------------------------------------------
# Registry entry shapes (mirror the YAMLs in configs/)
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    name: str
    family: str  # e.g. "gemma3n"
    hf_id: str
    multimodal: list[str] = Field(default_factory=lambda: ["text"])
    effective_params_b: float | None = None
    raw_params_b: float | None = None
    license: str | None = None
    gated: bool = False
    default_dtype: str = "bfloat16"
    notes: str | None = None


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    kind: str  # "instruction" | "vision_pair" | "audio_pair" | "knowledge_base"
    source: str  # "hf" | "local" | "synthetic" | "url"
    location: str
    splits: dict[str, int] = Field(default_factory=dict)
    license: str | None = None
    notes: str | None = None


class RecipeEntry(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    name: str
    backend: str = "hf_trl"  # "hf_trl" | "unsloth"
    method: str = "qlora"  # "qlora" | "lora" | "full"
    base_model: str  # ref to a ModelEntry.name
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    freeze_vision: bool = True
    freeze_audio: bool = True
    quantization: str = "nf4"
    max_seq_len: int = 4096
    per_device_batch_size: int = 1
    grad_accum: int = 16
    epochs: float = 2.0
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 200
    datasets: list[str] = Field(default_factory=list)
    notes: str | None = None


class TargetEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    kind: str  # "gguf" | "mediapipe" | "onnx" | "safetensors" | "container"
    platform: str | None = None  # "android" | "desktop" | "cloud:modal" | etc.
    quantization: str | None = None
    runtime: str | None = None  # "llama.cpp" | "mediapipe" | "vllm" | "ort"
    params: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


# ---------------------------------------------------------------------------
# Project + Run + Job
# ---------------------------------------------------------------------------


class ProjectSpec(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    description: str = ""
    base_model: str  # ref to ModelEntry.name
    datasets: list[str] = Field(default_factory=list)
    recipes: list[str] = Field(default_factory=list)
    targets: list[str] = Field(default_factory=list)
    eval_set: str | None = None
    system_prompt_file: str | None = None
    judge_prompt_file: str | None = None
    redteam_prompt_file: str | None = None
    notes: str | None = None


class RunSpec(BaseModel):
    """User-facing description of a run; the orchestrator expands this to jobs."""
    project: str
    recipe: str
    targets: list[str] = Field(default_factory=list)  # empty = all from project
    dataset_overrides: list[str] = Field(default_factory=list)
    compute: Compute = Compute.LOCAL_GPU
    dry_run: bool = False
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class JobMetric(BaseModel):
    name: str
    value: float
    unit: str | None = None


class JobRecord(BaseModel):
    """Snapshot of a job as the dashboard / API sees it."""
    id: str
    run_id: str
    project: str
    kind: JobKind
    status: JobStatus
    compute: Compute
    deps: list[str] = Field(default_factory=list)
    queue: str | None = None
    rq_job_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: list[JobMetric] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    logs_path: str | None = None


class RunRecord(BaseModel):
    id: str
    project: str
    recipe: str
    targets: list[str]
    compute: Compute
    status: JobStatus
    created_at: datetime
    finished_at: datetime | None = None
    notes: str | None = None
    job_ids: list[str] = Field(default_factory=list)
