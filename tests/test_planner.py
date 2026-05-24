from __future__ import annotations

from eat.orchestrator import planner
from eat.types import Compute, JobKind, RunSpec


def test_plan_food_qlora_text_includes_train_and_export():
    spec = RunSpec(project="food_health_coach", recipe="qlora_text",
                   targets=["gguf_q4km"], compute=Compute.LOCAL_GPU)
    run, jobs = planner.plan(spec)
    kinds = {j.kind for j in jobs}
    assert JobKind.TRAIN in kinds
    assert JobKind.EVAL_QUALITY in kinds
    assert JobKind.EVAL_SAFETY in kinds
    assert JobKind.EXPORT in kinds
    assert JobKind.BENCH_LOCAL in kinds
    train = next(j for j in jobs if j.kind == JobKind.TRAIN)
    assert train.compute == Compute.LOCAL_GPU


def test_plan_smoke_is_single_job():
    run, jobs = planner.plan_smoke()
    assert len(jobs) == 1
    assert jobs[0].kind == JobKind.SMOKE


def test_dag_no_cycles():
    spec = RunSpec(project="food_health_coach", recipe="qlora_vlm")
    run, jobs = planner.plan(spec)
    by_id = {j.id: j for j in jobs}
    # Each dep should resolve to a real job, and the deps should be acyclic.
    seen: set[str] = set()
    def _visit(jid: str, stack: set[str]) -> None:
        assert jid not in stack, f"cycle through {jid}"
        if jid in seen:
            return
        for d in by_id[jid].deps:
            _visit(d, stack | {jid})
        seen.add(jid)
    for j in jobs:
        _visit(j.id, set())
