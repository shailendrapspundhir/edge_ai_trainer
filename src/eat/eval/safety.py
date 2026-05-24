"""Safety / red-team eval.

Loads the project's `red_team.txt` (one JSON object per line — see the food
project for the schema), generates candidate answers, and checks the
`must` / `must_not` clauses.

The check is intentionally simple: substring + keyword matching. Replace with
an LLM-judge later when we want richer scoring; the gate (hard_fails) stays
binary either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eat.logging_setup import get_logger
from eat.paths import projects_dir, run_artifacts_dir
from eat.registry import projects as projects_reg

log = get_logger("eval.safety")


def _candidate(prompt: dict[str, Any], run_id: str) -> str:  # noqa: ARG001
    # Same placeholder as quality.py — replace with a real loader when eval moves to GPU queue.
    return "[candidate-answer-placeholder]"


def _check(text: str, must: str | None, must_not: str | None) -> tuple[bool, str]:
    text_l = (text or "").lower()
    if must and must.lower() not in text_l:
        return False, f"missing required: {must!r}"
    if must_not and must_not.lower() in text_l:
        return False, f"contains forbidden: {must_not!r}"
    return True, ""


def evaluate(run_id: str, project: str | None = None) -> dict[str, Any]:
    if not project:
        from eat.orchestrator import service
        try:
            run, _ = service.get_run(run_id)
            project = run.project
        except Exception:  # noqa: BLE001
            project = None

    rt_path: Path | None = None
    if project:
        proj = projects_reg.get(project)
        if proj.redteam_prompt_file:
            rt_path = projects_dir() / project / proj.redteam_prompt_file

    cases: list[dict[str, Any]] = []
    if rt_path and rt_path.exists():
        for line in rt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows: list[dict[str, Any]] = []
    hard_fails = 0
    for case in cases:
        cand = _candidate(case, run_id)
        ok, reason = _check(cand, case.get("must"), case.get("must_not"))
        if not ok:
            hard_fails += 1
        rows.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "candidate": cand,
            "pass": ok,
            "reason": reason,
        })
    n = len(rows) or 1
    pass_rate = sum(1 for r in rows if r["pass"]) / n
    out_dir = run_artifacts_dir(run_id)
    out = out_dir / "safety_report.json"
    out.write_text(json.dumps({
        "run_id": run_id,
        "project": project,
        "n_cases": len(rows),
        "pass_rate": pass_rate,
        "hard_fails": hard_fails,
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    return {"path": str(out), "pass_rate": pass_rate, "hard_fails": hard_fails}
