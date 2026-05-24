"""Job definitions — the actual callables that RQ workers execute.

Each function takes a `job_id`, fetches its own params from the DB, runs the
work, and writes results back. This keeps the queue payload tiny (just an id)
and means we can re-run a job by id alone.

The heavy modules (transformers, peft, llama.cpp wrappers) are imported lazily
inside the functions so the orchestrator + dashboard don't drag them into
their process.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from eat.config import get_settings
from eat.logging_setup import get_logger
from eat.orchestrator import db
from eat.paths import run_artifacts_dir, run_dir
from eat.types import JobKind, JobStatus

log = get_logger("jobs")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mark(job_id: str, status: JobStatus, **fields: Any) -> None:
    with db.session() as s:
        j = s.get(db.JobModel, job_id)
        if j is None:
            raise RuntimeError(f"job missing: {job_id}")
        j.status = status.value
        if status == JobStatus.RUNNING and not j.started_at:
            j.started_at = datetime.utcnow()
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.SKIPPED):
            j.finished_at = datetime.utcnow()
        for k, v in fields.items():
            setattr(j, k, v)


def _params(job_id: str) -> dict[str, Any]:
    with db.session() as s:
        j = s.get(db.JobModel, job_id)
        if j is None:
            raise RuntimeError(f"job missing: {job_id}")
        return db.loads(j.params_json, {})


def _add_metric(job_id: str, name: str, value: float, unit: str | None = None) -> None:
    with db.session() as s:
        j = s.get(db.JobModel, job_id)
        if j is None:
            return
        metrics = db.loads(j.metrics_json, [])
        metrics.append({"name": name, "value": value, "unit": unit})
        j.metrics_json = db.dumps(metrics)


def _add_artifact(job_id: str, name: str, path: str) -> None:
    with db.session() as s:
        j = s.get(db.JobModel, job_id)
        if j is None:
            return
        artifacts = db.loads(j.artifacts_json, {})
        artifacts[name] = path
        j.artifacts_json = db.dumps(artifacts)


def _run_id_for(job_id: str) -> str:
    with db.session() as s:
        j = s.get(db.JobModel, job_id)
        return j.run_id if j else ""


def _open_logs(job_id: str) -> Path:
    rid = _run_id_for(job_id)
    if not rid:
        return run_dir("orphan") / f"{job_id}.log"
    return run_dir(rid) / f"{job_id}.log"


def _wrap(job_id: str, kind: JobKind, fn) -> None:
    """Common wrapper: status transitions, logging, exception capture."""
    log_path = _open_logs(job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _mark(job_id, JobStatus.RUNNING, logs_path=str(log_path))
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== {datetime.utcnow().isoformat()} START {kind.value} ===\n")
            fh.flush()
            fn(fh)
        _mark(job_id, JobStatus.COMPLETED, error=None)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        log.error("job_failed", job_id=job_id, kind=kind.value, error=str(exc))
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== ERROR ===\n{tb}\n")
        _mark(job_id, JobStatus.FAILED, error=str(exc))


# ---------------------------------------------------------------------------
# the actual jobs
# ---------------------------------------------------------------------------


def run_smoke(job_id: str) -> None:
    """A no-op job used by `make smoke` to validate the pipeline end-to-end."""
    def _do(fh):
        params = _params(job_id)
        for i in range(int(params.get("steps", 5))):
            fh.write(f"smoke step {i}\n"); fh.flush()
            time.sleep(0.2)
        _add_metric(job_id, "fake_loss", 0.42, "loss")
        _add_artifact(job_id, "marker", str(run_artifacts_dir(_run_id_for(job_id)) / "smoke.txt"))
        Path(run_artifacts_dir(_run_id_for(job_id)) / "smoke.txt").write_text("ok\n")
    _wrap(job_id, JobKind.SMOKE, _do)


def prepare_data(job_id: str) -> None:
    def _do(fh):
        from eat.data import loaders
        params = _params(job_id)
        project = params["project"]
        dataset = params["dataset"]
        fh.write(f"prepare_data project={project} dataset={dataset}\n"); fh.flush()
        stats = loaders.prepare(project=project, dataset=dataset)
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                _add_metric(job_id, k, float(v))
            fh.write(f"{k}={v}\n")
    _wrap(job_id, JobKind.PREPARE_DATA, _do)


def train(job_id: str) -> None:
    def _do(fh):
        from eat.training import dispatch
        params = _params(job_id)
        result = dispatch.train(
            run_id=_run_id_for(job_id),
            project=params["project"],
            recipe=params["recipe"],
            log_file=fh,
        )
        for name, val in result.get("metrics", {}).items():
            _add_metric(job_id, name, float(val))
        for name, path in result.get("artifacts", {}).items():
            _add_artifact(job_id, name, path)
    _wrap(job_id, JobKind.TRAIN, _do)


def eval_quality(job_id: str) -> None:
    def _do(fh):
        from eat.eval import quality
        params = _params(job_id)
        report = quality.evaluate(_run_id_for(job_id), project=params.get("project"))
        for k, v in report.get("scores", {}).items():
            _add_metric(job_id, f"quality.{k}", float(v))
        _add_artifact(job_id, "quality_report", report["path"])
        fh.write(json.dumps(report.get("summary", {}), indent=2))
    _wrap(job_id, JobKind.EVAL_QUALITY, _do)


def eval_safety(job_id: str) -> None:
    def _do(fh):
        from eat.eval import safety
        params = _params(job_id)
        report = safety.evaluate(_run_id_for(job_id), project=params.get("project"))
        _add_metric(job_id, "safety.pass_rate", float(report["pass_rate"]))
        _add_metric(job_id, "safety.hard_fails", float(report["hard_fails"]))
        _add_artifact(job_id, "safety_report", report["path"])
    _wrap(job_id, JobKind.EVAL_SAFETY, _do)


def eval_perf(job_id: str) -> None:
    def _do(fh):
        from eat.eval import perf
        params = _params(job_id)
        report = perf.evaluate(
            _run_id_for(job_id),
            runtime=params.get("runtime", "llama.cpp"),
            quant=params.get("quant", "Q4_K_M"),
        )
        for k, v in report.get("metrics", {}).items():
            _add_metric(job_id, f"perf.{k}", float(v))
        _add_artifact(job_id, "perf_report", report["path"])
    _wrap(job_id, JobKind.EVAL_PERF, _do)


def quantize(job_id: str) -> None:
    def _do(fh):
        from eat.export import gguf
        params = _params(job_id)
        out = gguf.export(_run_id_for(job_id), quant=params.get("quant", "Q4_K_M"))
        _add_artifact(job_id, "gguf", str(out))
        fh.write(f"gguf={out}\n")
    _wrap(job_id, JobKind.QUANTIZE, _do)


def export(job_id: str) -> None:
    def _do(fh):
        from eat.export import dispatch
        params = _params(job_id)
        out = dispatch.export(_run_id_for(job_id), target=params["target"])
        _add_artifact(job_id, params["target"], str(out))
        fh.write(f"{params['target']}={out}\n")
    _wrap(job_id, JobKind.EXPORT, _do)


def bench_local(job_id: str) -> None:
    def _do(fh):
        from eat.eval import perf
        params = _params(job_id)
        report = perf.evaluate(
            _run_id_for(job_id),
            runtime=params.get("runtime", "llama.cpp"),
            quant=params.get("quant", "Q4_K_M"),
        )
        for k, v in report.get("metrics", {}).items():
            _add_metric(job_id, f"bench.{k}", float(v))
        _add_artifact(job_id, "bench_report", report["path"])
    _wrap(job_id, JobKind.BENCH_LOCAL, _do)


def bench_android(job_id: str) -> None:
    def _do(fh):
        from eat.android import bench
        params = _params(job_id)
        report = bench.run_bench(
            _run_id_for(job_id),
            target=params.get("target", "mediapipe_litert"),
            device_serial=params.get("device") or get_settings().android_device_serial,
        )
        for k, v in report.get("metrics", {}).items():
            _add_metric(job_id, f"android.{k}", float(v))
        _add_artifact(job_id, "android_report", report["path"])
    _wrap(job_id, JobKind.BENCH_ANDROID, _do)


def cloud_build(job_id: str) -> None:
    def _do(fh):
        from eat.cloud import containers
        params = _params(job_id)
        image = containers.build(_run_id_for(job_id), target=params["target"])
        _add_artifact(job_id, "image", image)
        fh.write(f"image={image}\n")
    _wrap(job_id, JobKind.CLOUD_BUILD, _do)


def cloud_deploy(job_id: str) -> None:
    def _do(fh):
        from eat.cloud import deploy
        params = _params(job_id)
        endpoint = deploy.deploy(_run_id_for(job_id), target=params["target"])
        _add_artifact(job_id, "endpoint", endpoint)
        fh.write(f"endpoint={endpoint}\n")
    _wrap(job_id, JobKind.CLOUD_DEPLOY, _do)


def cloud_bench(job_id: str) -> None:
    def _do(fh):
        from eat.cloud import bench_remote
        params = _params(job_id)
        result = bench_remote.bench(_run_id_for(job_id), endpoint=params.get("endpoint"))
        for k, v in result.get("metrics", {}).items():
            _add_metric(job_id, f"cloud.{k}", float(v))
        _add_artifact(job_id, "cloud_bench_report", result["path"])
    _wrap(job_id, JobKind.CLOUD_BENCH, _do)


# ---------------------------------------------------------------------------
# kind -> callable map (used by the queue dispatcher)
# ---------------------------------------------------------------------------


DISPATCH = {
    JobKind.SMOKE: run_smoke,
    JobKind.PREPARE_DATA: prepare_data,
    JobKind.TRAIN: train,
    JobKind.EVAL_QUALITY: eval_quality,
    JobKind.EVAL_SAFETY: eval_safety,
    JobKind.EVAL_PERF: eval_perf,
    JobKind.QUANTIZE: quantize,
    JobKind.EXPORT: export,
    JobKind.BENCH_LOCAL: bench_local,
    JobKind.BENCH_ANDROID: bench_android,
    JobKind.CLOUD_BUILD: cloud_build,
    JobKind.CLOUD_DEPLOY: cloud_deploy,
    JobKind.CLOUD_BENCH: cloud_bench,
}


def run_job(job_id: str, kind: str) -> None:
    """Entry point that RQ enqueues — dispatches to the right callable."""
    fn = DISPATCH[JobKind(kind)]
    fn(job_id)
