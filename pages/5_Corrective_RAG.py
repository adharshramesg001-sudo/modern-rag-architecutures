import streamlit as st

from rag_compare.architectures import corrective
from rag_compare.ui import render_architecture_page

st.set_page_config(page_title=f"{corrective.SPEC.name} · Modern RAG", page_icon=corrective.SPEC.icon, layout="wide")
render_architecture_page(corrective.SPEC)
