import streamlit as st

from rag_compare.architectures import graph
from rag_compare.ui import render_architecture_page

st.set_page_config(page_title=f"{graph.SPEC.name} · Modern RAG", page_icon=graph.SPEC.icon, layout="wide")
render_architecture_page(graph.SPEC)
