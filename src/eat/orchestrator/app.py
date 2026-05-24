"""FastAPI control plane — `eat-orchestrator`.

Endpoints:

    GET  /healthz
    GET  /runs?project=&limit=
    POST /runs                         body: RunSpec
    GET  /runs/{id}
    POST /runs/{id}/cancel
    GET  /runs/{id}/logs               server-sent events stream
    GET  /jobs/{id}
    GET  /registry/{kind}              kind in {models, datasets, recipes, targets, projects}
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from eat.config import get_settings
from eat.logging_setup import get_logger, setup_logging
from eat.orchestrator import db, service
from eat.registry import datasets, models, projects, recipes, targets
from eat.types import RunSpec

log = get_logger("orchestrator.app")

app = FastAPI(title="edge-ai-trainer orchestrator", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    setup_logging()
    db.init_db()
    log.info("orchestrator_start", settings=get_settings().model_dump(exclude_none=True))


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True}


@app.get("/runs")
def list_runs(project: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    return [r.model_dump(mode="json") for r in service.list_runs(project=project, limit=limit)]


@app.post("/runs")
def post_run(spec: RunSpec) -> dict[str, Any]:
    run = service.submit_run(spec)
    return run.model_dump(mode="json")


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        run, jobs = service.get_run(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {"run": run.model_dump(mode="json"), "jobs": [j.model_dump(mode="json") for j in jobs]}


@app.post("/runs/{run_id}/cancel")
def cancel(run_id: str) -> dict[str, Any]:
    try:
        service.cancel_run(run_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True}


@app.get("/runs/{run_id}/logs")
async def stream_logs(run_id: str) -> StreamingResponse:
    async def _gen():
        queue_: asyncio.Queue[str] = asyncio.Queue()

        def _sink(line: str) -> None:
            try:
                queue_.put_nowait(line + "\n")
            except asyncio.QueueFull:
                pass

        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, service.tail_logs, run_id, True, _sink, 0.5)
        try:
            while not task.done() or not queue_.empty():
                try:
                    chunk = await asyncio.wait_for(queue_.get(), timeout=0.5)
                    yield f"data: {chunk}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            task.cancel()

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/registry/{kind}")
def registry_get(kind: str) -> list[dict[str, Any]]:
    table = {
        "models": models.all_entries,
        "datasets": datasets.all_entries,
        "recipes": recipes.all_entries,
        "targets": targets.all_entries,
        "projects": projects.all_specs,
    }
    if kind not in table:
        raise HTTPException(404, f"unknown registry kind: {kind}")
    return [e.model_dump(mode="json") for e in table[kind]()]


def main() -> int:
    s = get_settings()
    uvicorn.run(
        "eat.orchestrator.app:app",
        host=s.orchestrator_host,
        port=s.orchestrator_port,
        log_level=s.log_level.lower(),
        reload=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
