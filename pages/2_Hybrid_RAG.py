import streamlit as st

from rag_compare.architectures import hybrid
from rag_compare.ui import render_architecture_page

st.set_page_config(page_title=f"{hybrid.SPEC.name} · Modern RAG", page_icon=hybrid.SPEC.icon, layout="wide")
render_architecture_page(hybrid.SPEC)
