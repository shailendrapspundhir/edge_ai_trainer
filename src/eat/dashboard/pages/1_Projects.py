"""Projects page."""

import streamlit as st

from eat.registry import projects

st.set_page_config(page_title="projects", layout="wide")
st.title("projects")

specs = projects.all_specs()
if not specs:
    st.info("No projects yet. `eat project init <name>` to scaffold one.")
else:
    for spec in specs:
        with st.container(border=True):
            st.markdown(f"### {spec.name}")
            st.caption(spec.description or "—")
            cols = st.columns(4)
            cols[0].markdown(f"**base model**\n\n`{spec.base_model}`")
            cols[1].markdown("**datasets**\n\n" + ", ".join(spec.datasets or ["—"]))
            cols[2].markdown("**recipes**\n\n" + ", ".join(spec.recipes or ["—"]))
            cols[3].markdown("**targets**\n\n" + ", ".join(spec.targets or ["—"]))
            if spec.notes:
                with st.expander("notes"):
                    st.write(spec.notes)
