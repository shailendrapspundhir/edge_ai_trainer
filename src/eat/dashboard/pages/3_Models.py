"""Models registry view."""

import streamlit as st

from eat.registry import models, recipes

st.set_page_config(page_title="models", layout="wide")
st.title("models")

entries = models.all_entries()
if not entries:
    st.info("No models registered yet.")
else:
    st.dataframe([
        {
            "name": e.name, "family": e.family, "hf_id": e.hf_id,
            "modes": ",".join(e.multimodal), "effective B": e.effective_params_b,
            "gated": e.gated, "license": e.license or "",
        } for e in entries
    ], use_container_width=True, hide_index=True)

st.divider()
st.subheader("recipes")
recs = recipes.all_entries()
if not recs:
    st.info("No recipes yet.")
else:
    st.dataframe([
        {"name": r.name, "backend": r.backend, "method": r.method,
         "base_model": r.base_model, "max_seq": r.max_seq_len,
         "lr": r.learning_rate, "epochs": r.epochs}
        for r in recs
    ], use_container_width=True, hide_index=True)
