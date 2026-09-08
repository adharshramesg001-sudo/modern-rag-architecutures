import streamlit as st

from rag_compare.architectures import classic
from rag_compare.ui import render_architecture_page

st.set_page_config(page_title=f"{classic.SPEC.name} · Modern RAG", page_icon=classic.SPEC.icon, layout="wide")
render_architecture_page(classic.SPEC)
