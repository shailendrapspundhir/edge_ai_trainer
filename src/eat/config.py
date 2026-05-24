"""Centralised settings.

Environment-first, with sensible defaults. Settings are loaded once on import of
`get_settings()` so reading them is cheap. Override anything via `.env` or by
exporting environment variables before launching a process.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration in one place."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General ---------------------------------------------------------------
    env: str = Field("local", alias="EAT_ENV")
    log_level: str = Field("INFO", alias="EAT_LOG_LEVEL")
    artifacts_dir: Path = Field(Path("./artifacts"), alias="EAT_ARTIFACTS_DIR")
    runs_dir: Path = Field(Path("./runs"), alias="EAT_RUNS_DIR")
    cache_dir: Path = Field(Path("./.cache/eat"), alias="EAT_CACHE_DIR")

    # HF --------------------------------------------------------------------
    hf_token: str | None = Field(None, alias="HUGGING_FACE_HUB_TOKEN")
    transformers_offline: bool = Field(False, alias="TRANSFORMERS_OFFLINE")

    # Orchestrator ----------------------------------------------------------
    orchestrator_host: str = Field("127.0.0.1", alias="EAT_ORCHESTRATOR_HOST")
    orchestrator_port: int = Field(8765, alias="EAT_ORCHESTRATOR_PORT")
    db_url: str = Field("sqlite:///./eat.sqlite", alias="EAT_DB_URL")
    redis_url: str = Field("redis://127.0.0.1:6379/0", alias="EAT_REDIS_URL")
    queue_gpu: str = Field("eat.gpu", alias="EAT_QUEUE_GPU")
    queue_cpu: str = Field("eat.cpu", alias="EAT_QUEUE_CPU")
    queue_android: str = Field("eat.android", alias="EAT_QUEUE_ANDROID")
    queue_cloud: str = Field("eat.cloud", alias="EAT_QUEUE_CLOUD")
    gpu_workers: int = Field(1, alias="EAT_GPU_WORKERS")
    cpu_workers: int = Field(4, alias="EAT_CPU_WORKERS")
    android_workers: int = Field(1, alias="EAT_ANDROID_WORKERS")
    worker_queues: str | None = Field(None, alias="EAT_WORKER_QUEUES")

    # Dashboard -------------------------------------------------------------
    dashboard_host: str = Field("127.0.0.1", alias="EAT_DASHBOARD_HOST")
    dashboard_port: int = Field(8501, alias="EAT_DASHBOARD_PORT")

    # MLflow ----------------------------------------------------------------
    mlflow_tracking_uri: str = Field("file:./mlruns", alias="MLFLOW_TRACKING_URI")
    mlflow_artifact_location: str | None = Field(None, alias="MLFLOW_ARTIFACT_LOCATION")
    mlflow_default_experiment: str = Field("edge-ai-trainer", alias="MLFLOW_EXPERIMENT_DEFAULT")

    # Teacher (data gen) ----------------------------------------------------
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    teacher_provider: str = Field("anthropic", alias="EAT_TEACHER_PROVIDER")
    teacher_model: str = Field("claude-opus-4-7", alias="EAT_TEACHER_MODEL")
    teacher_max_usd: float = Field(25.0, alias="EAT_TEACHER_MAX_USD")

    # Judge (eval) ----------------------------------------------------------
    judge_provider: str = Field("anthropic", alias="EAT_JUDGE_PROVIDER")
    judge_model: str = Field("claude-sonnet-4-6", alias="EAT_JUDGE_MODEL")
    judge_max_usd: float = Field(10.0, alias="EAT_JUDGE_MAX_USD")

    # Android ---------------------------------------------------------------
    adb_bin: str = Field("adb", alias="ADB_BIN")
    android_device_serial: str | None = Field(None, alias="EAT_ANDROID_DEVICE_SERIAL")
    android_remote_dir: str = Field("/data/local/tmp/eat", alias="EAT_ANDROID_REMOTE_DIR")
    android_app_id: str = Field("com.eat.bench", alias="EAT_ANDROID_APP_ID")
    android_bench_timeout_s: int = Field(600, alias="EAT_ANDROID_BENCH_TIMEOUT_S")

    # Cloud serving ---------------------------------------------------------
    container_registry: str = Field("ghcr.io/your-org/eat", alias="EAT_CONTAINER_REGISTRY")
    default_cloud_platform: str = Field("modal", alias="EAT_DEFAULT_CLOUD_PLATFORM")
    modal_token_id: str | None = Field(None, alias="MODAL_TOKEN_ID")
    modal_token_secret: str | None = Field(None, alias="MODAL_TOKEN_SECRET")
    runpod_api_key: str | None = Field(None, alias="RUNPOD_API_KEY")
    fly_api_token: str | None = Field(None, alias="FLY_API_TOKEN")

    artifact_s3_bucket: str | None = Field(None, alias="EAT_ARTIFACT_S3_BUCKET")
    aws_access_key_id: str | None = Field(None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(None, alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str | None = Field(None, alias="AWS_DEFAULT_REGION")

    # Cloud training --------------------------------------------------------
    skypilot_default_cloud: str = Field("runpod", alias="EAT_SKYPILOT_DEFAULT_CLOUD")
    cloud_train_max_usd: float = Field(50.0, alias="EAT_CLOUD_TRAIN_MAX_USD")
    cloud_train_max_hours: float = Field(24.0, alias="EAT_CLOUD_TRAIN_MAX_HOURS")

    # DVC -------------------------------------------------------------------
    dvc_remote: str | None = Field(None, alias="EAT_DVC_REMOTE")

    def queue_names(self) -> list[str]:
        return [self.queue_gpu, self.queue_cpu, self.queue_android, self.queue_cloud]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide singleton Settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Drop the cache and re-read settings. Useful in tests."""
    get_settings.cache_clear()
    return get_settings()
