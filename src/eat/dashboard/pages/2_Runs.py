"""Runs page — list, drill into a single run, tail logs."""

from pathlib import Path

import streamlit as st

from eat.orchestrator import service
from eat.paths import run_dir
from eat.registry import projects, recipes


def _tail(path: Path, n: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


st.set_page_config(page_title="runs", layout="wide")
st.title("runs")

with st.sidebar:
    st.subheader("filter")
    proj_filter = st.selectbox("project", ["(all)", *[p.name for p in projects.all_specs()]])
    limit = st.slider("limit", 5, 200, 50)

runs = service.list_runs(
    project=None if proj_filter == "(all)" else proj_filter,
    limit=limit,
)

if not runs:
    st.info("No runs match the current filter.")
else:
    chosen = st.selectbox(
        "run", [r.id for r in runs],
        format_func=lambda rid: next((f"{r.id} — {r.project}/{r.recipe} [{r.status.value}]"
                                      for r in runs if r.id == rid), rid),
    )
    if chosen:
        run, jobs = service.get_run(chosen)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("status", run.status.value)
        c2.metric("compute", run.compute.value)
        c3.metric("jobs", len(jobs))
        c4.metric("targets", len(run.targets))

        st.subheader("jobs")
        st.dataframe([
            {
                "id": j.id, "kind": j.kind.value, "status": j.status.value,
                "queue": j.queue or "", "started": str(j.started_at or ""),
                "finished": str(j.finished_at or ""), "error": (j.error or "")[:80],
            } for j in jobs
        ], use_container_width=True, hide_index=True)

        st.subheader("metrics")
        metric_rows = []
        for j in jobs:
            for m in j.metrics:
                metric_rows.append({"job": j.kind.value, "metric": m.name,
                                    "value": m.value, "unit": m.unit or ""})
        if metric_rows:
            st.dataframe(metric_rows, use_container_width=True, hide_index=True)
        else:
            st.caption("no metrics yet")

        st.subheader("logs (tail)")
        folder = run_dir(run.id)
        log_files = sorted(folder.glob("*.log"))
        if not log_files:
            st.caption("no logs yet")
        else:
            for f in log_files:
                with st.expander(f.name):
                    st.code(_tail(f, 200), language="text")
