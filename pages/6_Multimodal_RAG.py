import streamlit as st

from rag_compare.architectures import multimodal
from rag_compare.ui import render_architecture_page

st.set_page_config(page_title=f"{multimodal.SPEC.name} · Modern RAG", page_icon=multimodal.SPEC.icon, layout="wide")
render_architecture_page(multimodal.SPEC)
