import pandas as pd
import streamlit as st

from rag_compare.registry import ALL_SPECS
from rag_compare.ui import inject_css, render_pipeline, render_sidebar_llm_config

st.set_page_config(page_title="Compare All · Modern RAG", page_icon="📊", layout="wide")
inject_css()
render_sidebar_llm_config()

st.title("📊 Compare All Architectures")
st.caption("Every architecture pulls from the same retrieval stack — the difference is what each one does with it.")

dimensions = list(ALL_SPECS[0].compare_row.keys())
table = {"Dimension": dimensions}
for spec in ALL_SPECS:
    table[f"{spec.icon} {spec.name}"] = [spec.compare_row[d] for d in dimensions]

df = pd.DataFrame(table).set_index("Dimension")
st.dataframe(df, use_container_width=True)

st.markdown("---")
st.subheader("Pipelines at a glance")

for spec in ALL_SPECS:
    st.markdown(f"**{spec.icon} {spec.name} — {spec.tagline}**")
    render_pipeline(spec.pipeline)

st.markdown("---")
st.markdown(
    '<div class="callout">💡 <strong>These patterns compose.</strong> An agentic '
    "workflow can use hybrid search, retrieve graph context, inspect a chart, and "
    "evaluate the evidence before answering. Every addition also introduces "
    "latency, cost, and more behavior to evaluate — pick the smallest architecture "
    "that solves the problem in front of you.</div>",
    unsafe_allow_html=True,
)
