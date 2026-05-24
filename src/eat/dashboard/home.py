"""Edge AI Trainer — dashboard home page."""

from __future__ import annotations

import streamlit as st

from eat.config import get_settings
from eat.orchestrator import db, service

st.set_page_config(page_title="edge-ai-trainer", page_icon=":bar_chart:", layout="wide")

db.init_db()
s = get_settings()

st.title("edge-ai-trainer")
st.caption("Local-first workbench for training, evaluating, and deploying edge AI models.")

col1, col2, col3, col4 = st.columns(4)
runs = service.list_runs(limit=200)
col1.metric("runs", len(runs))
col2.metric("running", sum(1 for r in runs if r.status.value == "running"))
col3.metric("completed", sum(1 for r in runs if r.status.value == "completed"))
col4.metric("failed", sum(1 for r in runs if r.status.value == "failed"))

st.subheader("recent runs")
if not runs:
    st.info("No runs yet. Submit one: `eat run --project food_health_coach --recipe smoke`")
else:
    rows = [
        {"id": r.id, "project": r.project, "recipe": r.recipe,
         "status": r.status.value, "compute": r.compute.value,
         "created": r.created_at.isoformat(timespec="seconds")}
        for r in runs[:50]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.divider()
st.subheader("environment")
st.json({
    "orchestrator": f"{s.orchestrator_host}:{s.orchestrator_port}",
    "redis": s.redis_url,
    "db": s.db_url,
    "mlflow": s.mlflow_tracking_uri,
    "artifacts_dir": str(s.artifacts_dir),
}, expanded=False)
