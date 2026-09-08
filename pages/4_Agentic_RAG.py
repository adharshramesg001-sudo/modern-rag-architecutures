import streamlit as st

from rag_compare.architectures import agentic
from rag_compare.ui import render_architecture_page

st.set_page_config(page_title=f"{agentic.SPEC.name} · Modern RAG", page_icon=agentic.SPEC.icon, layout="wide")
render_architecture_page(agentic.SPEC)
