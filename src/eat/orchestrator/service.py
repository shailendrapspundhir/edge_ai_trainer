"""Public service layer used by the CLI, FastAPI, and dashboard.

This module owns all DB writes related to runs/jobs. Workers update job state
via `eat.orchestrator.jobs` (which writes through the same db helpers).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from eat.logging_setup import get_logger
from eat.orchestrator import db, planner, queue
from eat.paths import run_dir
from eat.types import (
    Compute,
    JobKind,
    JobMetric,
    JobRecord,
    JobStatus,
    RunRecord,
    RunSpec,
)

log = get_logger("service")


# ---------------------------------------------------------------------------
# Record <-> ORM converters
# ---------------------------------------------------------------------------


def _job_to_orm(j: JobRecord) -> db.JobModel:
    return db.JobModel(
        id=j.id,
        run_id=j.run_id,
        project=j.project,
        kind=j.kind.value,
        status=j.status.value,
        compute=j.compute.value,
        queue=j.queue,
        rq_job_id=j.rq_job_id,
        deps_json=db.dumps(j.deps),
        params_json=db.dumps(j.params),
        artifacts_json=db.dumps(j.artifacts),
        metrics_json=db.dumps([m.model_dump() for m in j.metrics]),
        logs_path=j.logs_path,
        error=j.error,
        created_at=j.created_at,
        started_at=j.started_at,
        finished_at=j.finished_at,
    )


def _orm_to_job(m: db.JobModel) -> JobRecord:
    return JobRecord(
        id=m.id,
        run_id=m.run_id,
        project=m.project,
        kind=JobKind(m.kind),
        status=JobStatus(m.status),
        compute=Compute(m.compute),
        queue=m.queue,
        rq_job_id=m.rq_job_id,
        deps=db.loads(m.deps_json, []),
        params=db.loads(m.params_json, {}),
        artifacts=db.loads(m.artifacts_json, {}),
        metrics=[JobMetric(**x) for x in db.loads(m.metrics_json, [])],
        logs_path=m.logs_path,
        error=m.error,
        created_at=m.created_at,
        started_at=m.started_at,
        finished_at=m.finished_at,
    )


def _run_to_orm(r: RunRecord) -> db.RunModel:
    return db.RunModel(
        id=r.id,
        project=r.project,
        recipe=r.recipe,
        targets_json=db.dumps(r.targets),
        compute=r.compute.value,
        status=r.status.value,
        notes=r.notes,
        created_at=r.created_at,
        finished_at=r.finished_at,
    )


def _orm_to_run(m: db.RunModel) -> RunRecord:
    return RunRecord(
        id=m.id,
        project=m.project,
        recipe=m.recipe,
        targets=db.loads(m.targets_json, []),
        compute=Compute(m.compute),
        status=JobStatus(m.status),
        created_at=m.created_at,
        finished_at=m.finished_at,
        notes=m.notes,
        job_ids=[j.id for j in m.jobs],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def submit_run(spec: RunSpec) -> RunRecord:
    db.init_db()
    if spec.project == "_smoke":
        run, jobs = planner.plan_smoke(notes=spec.notes or "")
    else:
        run, jobs = planner.plan(spec)

    with db.session() as s:
        s.add(_run_to_orm(run))
        for j in jobs:
            s.add(_job_to_orm(j))

    if spec.dry_run:
        log.info("dry_run_planned", run_id=run.id, jobs=len(jobs))
        return run

    _enqueue_all(run, jobs)
    return run


def _enqueue_all(run: RunRecord, jobs: list[JobRecord]) -> None:
    id_to_job: dict[str, JobRecord] = {j.id: j for j in jobs}
    rq_ids: dict[str, str] = {}
    # topo: parents-first via memoized recursion
    visited: set[str] = set()

    def _enq(jid: str) -> str:
        if jid in rq_ids:
            return rq_ids[jid]
        if jid in visited:
            raise RuntimeError(f"cycle detected on {jid}")
        visited.add(jid)
        j = id_to_job[jid]
        dep_rq_ids = [_enq(d) for d in j.deps]
        qname = queue.queue_for(j.kind, j.compute)
        rq_id = queue.enqueue(qname, j.id, j.kind, depends_on=dep_rq_ids or None)
        rq_ids[jid] = rq_id
        with db.session() as s:
            m = s.get(db.JobModel, j.id)
            if m:
                m.queue = qname
                m.rq_job_id = rq_id
                m.status = JobStatus.QUEUED.value
        return rq_id

    for j in jobs:
        _enq(j.id)
    log.info("run_enqueued", run_id=run.id, jobs=len(jobs))


def get_run(run_id: str) -> tuple[RunRecord, list[JobRecord]]:
    db.init_db()
    with db.session() as s:
        m = s.get(db.RunModel, run_id)
        if m is None:
            raise KeyError(f"run not found: {run_id}")
        return _orm_to_run(m), [_orm_to_job(j) for j in m.jobs]


def list_runs(project: str | None = None, limit: int = 20) -> list[RunRecord]:
    db.init_db()
    with db.session() as s:
        q = s.query(db.RunModel)
        if project:
            q = q.filter(db.RunModel.project == project)
        rows = q.order_by(db.RunModel.created_at.desc()).limit(limit).all()
        return [_orm_to_run(r) for r in rows]


def cancel_run(run_id: str) -> None:
    db.init_db()
    with db.session() as s:
        m = s.get(db.RunModel, run_id)
        if m is None:
            raise KeyError(run_id)
        m.status = JobStatus.CANCELLED.value
        m.finished_at = datetime.utcnow()
        for j in m.jobs:
            if j.status in (JobStatus.PENDING.value, JobStatus.QUEUED.value):
                j.status = JobStatus.CANCELLED.value
                # best-effort RQ cancel
                if j.rq_job_id:
                    try:
                        from rq.job import Job
                        Job.fetch(j.rq_job_id, connection=queue.get_redis()).cancel()
                    except Exception:  # noqa: BLE001
                        pass


def tail_logs(
    run_id: str,
    follow: bool = False,
    sink: Callable[[str], None] = print,
    poll_s: float = 0.5,
) -> None:
    """Tail every per-job log file under runs/<run_id>/."""
    db.init_db()
    folder = run_dir(run_id)
    offsets: dict[str, int] = {}
    try:
        while True:
            for log_path in sorted(folder.glob("*.log")):
                off = offsets.get(log_path.name, 0)
                with log_path.open("r", encoding="utf-8") as fh:
                    fh.seek(off)
                    chunk = fh.read()
                    offsets[log_path.name] = fh.tell()
                if chunk:
                    for line in chunk.splitlines():
                        sink(f"[{log_path.stem}] {line}")
            if not follow:
                return
            _, jobs = get_run(run_id)
            done = all(j.status in (JobStatus.COMPLETED, JobStatus.FAILED,
                                    JobStatus.CANCELLED, JobStatus.SKIPPED) for j in jobs)
            if done:
                return
            time.sleep(poll_s)
    except KeyboardInterrupt:
        return


def refresh_run_status(run_id: str) -> RunRecord:
    """Recompute run status from its jobs (called by API + dashboard)."""
    db.init_db()
    with db.session() as s:
        m = s.get(db.RunModel, run_id)
        if m is None:
            raise KeyError(run_id)
        statuses = [JobStatus(j.status) for j in m.jobs]
        if all(s_ == JobStatus.COMPLETED for s_ in statuses):
            m.status = JobStatus.COMPLETED.value
            m.finished_at = m.finished_at or datetime.utcnow()
        elif any(s_ == JobStatus.FAILED for s_ in statuses):
            m.status = JobStatus.FAILED.value
        elif any(s_ == JobStatus.RUNNING for s_ in statuses):
            m.status = JobStatus.RUNNING.value
        elif all(s_ in (JobStatus.PENDING, JobStatus.QUEUED) for s_ in statuses):
            m.status = JobStatus.QUEUED.value
        return _orm_to_run(m)
