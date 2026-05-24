"""Targets + per-run leaderboard."""

import streamlit as st

from eat.orchestrator import service
from eat.registry import targets

st.set_page_config(page_title="targets", layout="wide")
st.title("targets")

entries = targets.all_entries()
if not entries:
    st.info("No targets registered.")
else:
    st.dataframe([
        {"name": e.name, "kind": e.kind, "platform": e.platform or "",
         "runtime": e.runtime or "", "quantization": e.quantization or ""}
        for e in entries
    ], use_container_width=True, hide_index=True)

st.divider()
st.subheader("leaderboard — perf metrics across runs")
runs = service.list_runs(limit=100)
rows = []
for r in runs:
    _, jobs = service.get_run(r.id)
    for j in jobs:
        for m in j.metrics:
            if m.name.startswith(("perf.", "bench.", "android.", "cloud.")):
                rows.append({
                    "run": r.id, "project": r.project, "kind": j.kind.value,
                    "target": j.params.get("target", ""), "metric": m.name,
                    "value": m.value, "unit": m.unit or "",
                })
if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.caption("no perf metrics yet — submit a run and let the bench jobs complete")
