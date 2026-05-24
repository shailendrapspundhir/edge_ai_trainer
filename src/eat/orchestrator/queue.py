"""Thin RQ wrappers — single redis connection, queue picker."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import redis
from rq import Queue

from eat.config import get_settings
from eat.types import Compute, JobKind


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    return redis.from_url(get_settings().redis_url)


@lru_cache(maxsize=8)
def get_queue(name: str) -> Queue:
    return Queue(name, connection=get_redis())


def queue_for(kind: JobKind, compute: Compute) -> str:
    s = get_settings()
    if kind == JobKind.BENCH_ANDROID:
        return s.queue_android
    if kind in (JobKind.CLOUD_BUILD, JobKind.CLOUD_DEPLOY, JobKind.CLOUD_BENCH):
        return s.queue_cloud
    if kind == JobKind.TRAIN and compute == Compute.LOCAL_GPU:
        return s.queue_gpu
    if kind in (JobKind.EVAL_QUALITY, JobKind.EVAL_PERF, JobKind.BENCH_LOCAL) and compute == Compute.LOCAL_GPU:
        # GPU-eligible CPU work — keep on CPU queue unless explicitly on GPU
        return s.queue_cpu
    if compute == Compute.LOCAL_GPU:
        return s.queue_gpu
    return s.queue_cpu


def resolve_queues(spec: str | None) -> list[str]:
    """Resolve EAT_WORKER_QUEUES (comma-separated) into a queue-name list."""
    s = get_settings()
    if not spec:
        return s.queue_names()
    return [q.strip() for q in spec.split(",") if q.strip()]


def enqueue(
    queue_name: str,
    job_id: str,
    kind: JobKind,
    depends_on: Iterable[str] | None = None,
    timeout: int = 24 * 3600,
) -> str:
    """Enqueue a `run_job` call; returns the RQ job id."""
    from eat.orchestrator.jobs import run_job
    q = get_queue(queue_name)
    deps = list(depends_on) if depends_on else None
    rq_job = q.enqueue(
        run_job,
        args=(job_id, kind.value),
        job_id=f"rq-{job_id}",
        depends_on=deps,
        job_timeout=timeout,
        result_ttl=86400 * 7,
        failure_ttl=86400 * 30,
        description=f"{kind.value}:{job_id}",
    )
    return rq_job.id
