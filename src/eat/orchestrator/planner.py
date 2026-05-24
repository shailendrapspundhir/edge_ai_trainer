"""Turn a RunSpec into a DAG of JobRecord stubs."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from eat.registry import projects, recipes, targets
from eat.types import (
    Compute,
    JobKind,
    JobRecord,
    JobStatus,
    RunRecord,
    RunSpec,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def plan(spec: RunSpec) -> tuple[RunRecord, list[JobRecord]]:
    proj = projects.get(spec.project)
    recipe = recipes.get(spec.recipe)
    target_names = spec.targets or proj.targets or []
    target_entries = [targets.get(t) for t in target_names]

    run_id = _new_id("run")
    now = datetime.utcnow()

    # ---- dataset prep ----
    # Recipe-level `datasets` (or a CLI override) wins over the project's
    # default list — that's how recipes like `smoke` pull their own tiny
    # dataset instead of the project's real training corpus.
    train_ds_names = list(
        spec.dataset_overrides
        or getattr(recipe, "datasets", None)
        or proj.datasets
        or []
    )
    prep_ds_names: list[str] = list(train_ds_names)
    if proj.eval_set and proj.eval_set not in prep_ds_names:
        prep_ds_names.append(proj.eval_set)

    prep_jobs: list[JobRecord] = []
    for ds_name in prep_ds_names:
        prep_jobs.append(JobRecord(
            id=_new_id("job"),
            run_id=run_id,
            project=spec.project,
            kind=JobKind.PREPARE_DATA,
            status=JobStatus.PENDING,
            compute=Compute.LOCAL_CPU,
            deps=[],
            created_at=now,
            params={"project": spec.project, "dataset": ds_name},
        ))

    # ---- train ----
    train_job = JobRecord(
        id=_new_id("job"),
        run_id=run_id,
        project=spec.project,
        kind=JobKind.TRAIN,
        status=JobStatus.PENDING,
        compute=spec.compute,
        deps=[j.id for j in prep_jobs],
        created_at=now,
        params={"project": spec.project, "recipe": spec.recipe},
    )

    # ---- quality + safety eval (depend on train) ----
    eval_quality_job = JobRecord(
        id=_new_id("job"),
        run_id=run_id,
        project=spec.project,
        kind=JobKind.EVAL_QUALITY,
        status=JobStatus.PENDING,
        compute=Compute.LOCAL_CPU,
        deps=[train_job.id],
        created_at=now,
        params={"project": spec.project},
    )
    eval_safety_job = JobRecord(
        id=_new_id("job"),
        run_id=run_id,
        project=spec.project,
        kind=JobKind.EVAL_SAFETY,
        status=JobStatus.PENDING,
        compute=Compute.LOCAL_CPU,
        deps=[train_job.id],
        created_at=now,
        params={"project": spec.project},
    )

    # ---- per-target export + bench ----
    target_jobs: list[JobRecord] = []
    for t in target_entries:
        export_job = JobRecord(
            id=_new_id("job"),
            run_id=run_id,
            project=spec.project,
            kind=JobKind.EXPORT,
            status=JobStatus.PENDING,
            compute=Compute.LOCAL_CPU,
            deps=[train_job.id],
            created_at=now,
            params={"target": t.name, "project": spec.project},
        )
        target_jobs.append(export_job)
        if t.kind == "gguf":
            target_jobs.append(JobRecord(
                id=_new_id("job"),
                run_id=run_id,
                project=spec.project,
                kind=JobKind.BENCH_LOCAL,
                status=JobStatus.PENDING,
                compute=Compute.LOCAL_CPU,
                deps=[export_job.id],
                created_at=now,
                params={"target": t.name, "quant": t.quantization or "Q4_K_M", "runtime": "llama.cpp"},
            ))
        if t.platform == "android" or t.kind == "mediapipe":
            target_jobs.append(JobRecord(
                id=_new_id("job"),
                run_id=run_id,
                project=spec.project,
                kind=JobKind.BENCH_ANDROID,
                status=JobStatus.PENDING,
                compute=Compute.LOCAL_ANDROID,
                deps=[export_job.id],
                created_at=now,
                params={"target": t.name},
            ))
        if (t.platform or "").startswith("cloud"):
            build = JobRecord(
                id=_new_id("job"),
                run_id=run_id,
                project=spec.project,
                kind=JobKind.CLOUD_BUILD,
                status=JobStatus.PENDING,
                compute=Compute.LOCAL_CPU,
                deps=[export_job.id],
                created_at=now,
                params={"target": t.name},
            )
            depl = JobRecord(
                id=_new_id("job"),
                run_id=run_id,
                project=spec.project,
                kind=JobKind.CLOUD_DEPLOY,
                status=JobStatus.PENDING,
                compute=Compute.CLOUD_CPU,
                deps=[build.id],
                created_at=now,
                params={"target": t.name},
            )
            bench = JobRecord(
                id=_new_id("job"),
                run_id=run_id,
                project=spec.project,
                kind=JobKind.CLOUD_BENCH,
                status=JobStatus.PENDING,
                compute=Compute.CLOUD_CPU,
                deps=[depl.id],
                created_at=now,
                params={"target": t.name},
            )
            target_jobs.extend([build, depl, bench])

    all_jobs: list[JobRecord] = [*prep_jobs, train_job, eval_quality_job, eval_safety_job, *target_jobs]

    run = RunRecord(
        id=run_id,
        project=spec.project,
        recipe=spec.recipe,
        targets=target_names,
        compute=spec.compute,
        status=JobStatus.PENDING,
        created_at=now,
        notes=spec.notes,
        job_ids=[j.id for j in all_jobs],
    )
    return run, all_jobs


def plan_smoke(notes: str = "") -> tuple[RunRecord, list[JobRecord]]:
    """Minimal 1-job run used by `make smoke` / tests."""
    run_id = _new_id("run")
    now = datetime.utcnow()
    job = JobRecord(
        id=_new_id("job"),
        run_id=run_id,
        project="_smoke",
        kind=JobKind.SMOKE,
        status=JobStatus.PENDING,
        compute=Compute.LOCAL_CPU,
        deps=[],
        created_at=now,
        params={"steps": 3},
    )
    run = RunRecord(
        id=run_id,
        project="_smoke",
        recipe="smoke",
        targets=[],
        compute=Compute.LOCAL_CPU,
        status=JobStatus.PENDING,
        created_at=now,
        notes=notes or "smoke",
        job_ids=[job.id],
    )
    return run, [job]
