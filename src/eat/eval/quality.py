"""LLM-as-judge quality eval.

Generates the candidate's answer to each held-out eval prompt, then asks the
configured judge model to score on a fixed rubric. Writes a JSON report under
`artifacts/<run_id>/quality_report.json` and returns a summary.

If neither a judge API key nor a local candidate runner is available, the eval
falls back to a deterministic placeholder so the dashboard always has rows.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from eat.config import get_settings
from eat.data.loaders import iter_jsonl
from eat.logging_setup import get_logger
from eat.paths import projects_dir, run_artifacts_dir
from eat.registry import projects as projects_reg

log = get_logger("eval.quality")


def _candidate_answer(prompt_text: str, run_id: str) -> str:
    """Best-effort: load the merged model from the run and generate.

    For v0 we just stub — wiring a transformer generation call requires the
    `train` extra to be installed, which the eval worker (CPU queue) usually
    does not have. We return a placeholder; replace with a real loader when
    the eval is moved onto the GPU queue.
    """
    return "[candidate-answer-placeholder]"


def _judge_score(prompt_text: str, candidate: str, expected: dict[str, Any],
                 judge_prompt: str) -> dict[str, Any]:
    s = get_settings()
    if s.judge_provider == "anthropic" and s.anthropic_api_key:
        try:
            from anthropic import Anthropic  # type: ignore
            client = Anthropic(api_key=s.anthropic_api_key)
            resp = client.messages.create(
                model=s.judge_model,
                max_tokens=400,
                system=judge_prompt,
                messages=[{"role": "user", "content":
                    f"Prompt:\n{prompt_text}\n\nCandidate:\n{candidate}\n\n"
                    f"Expected hints (JSON):\n{json.dumps(expected, ensure_ascii=False)}"}],
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            try:
                return json.loads(text[text.find("{"): text.rfind("}") + 1])
            except json.JSONDecodeError:
                return {"raw": text, "parsed": False}
        except Exception as exc:  # noqa: BLE001
            log.warning("judge_api_failed", error=str(exc))

    # Fallback: deterministic zeros so reports remain valid JSON.
    return {
        "verdict_correctness": 0,
        "profile_faithfulness": 0,
        "hard_fail": False,
        "hard_fail_reason": "",
        "numerical_sanity": 0,
        "tone": 0,
        "explanation": "no judge configured",
    }


def evaluate(run_id: str, project: str | None = None) -> dict[str, Any]:
    out_dir = run_artifacts_dir(run_id)
    if not project:
        # try to read it from train_summary
        info = out_dir / "train_summary.json"
        if info.exists():
            project = json.loads(info.read_text()).get("project")
    if not project:
        project = _infer_project_from_run(run_id)

    proj = projects_reg.get(project) if project else None
    judge_prompt = ""
    eval_records: list[dict[str, Any]] = []

    if proj:
        if proj.judge_prompt_file:
            jp = projects_dir() / proj.name / proj.judge_prompt_file
            if jp.exists():
                judge_prompt = jp.read_text(encoding="utf-8")
        if proj.eval_set:
            from eat.registry import datasets as datasets_reg
            ds = datasets_reg.get(proj.eval_set)
            eval_records = list(iter_jsonl(Path(ds.location)))

    rows: list[dict[str, Any]] = []
    for ex in eval_records:
        prompt_text = json.dumps({"profile": ex.get("user_profile"),
                                  "food": ex.get("food"),
                                  "context": ex.get("context")}, ensure_ascii=False)
        candidate = _candidate_answer(prompt_text, run_id)
        score = _judge_score(prompt_text, candidate, ex, judge_prompt)
        rows.append({"id": ex.get("id"), "candidate": candidate, "score": score})

    axes = ("verdict_correctness", "profile_faithfulness", "numerical_sanity", "tone")
    scores: dict[str, float] = {}
    if rows:
        for a in axes:
            vals = [float(r["score"].get(a, 0) or 0) for r in rows]
            scores[a] = statistics.mean(vals) if vals else 0.0
        scores["hard_fail_rate"] = sum(1 for r in rows if r["score"].get("hard_fail")) / len(rows)
    else:
        scores = {a: 0.0 for a in axes} | {"hard_fail_rate": 0.0}

    summary = {
        "run_id": run_id,
        "project": project,
        "n_eval": len(rows),
        "scores": scores,
    }
    report = {"summary": summary, "rows": rows}
    out = out_dir / "quality_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {"path": str(out), "summary": summary, "scores": scores}


def _infer_project_from_run(run_id: str) -> str | None:
    from eat.orchestrator import service
    try:
        run, _ = service.get_run(run_id)
        return run.project
    except Exception:  # noqa: BLE001
        return None
