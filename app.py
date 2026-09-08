import streamlit as st

from rag_compare.registry import ALL_SPECS
from rag_compare.ui import inject_css, render_sidebar_api_key

st.set_page_config(page_title="Modern RAG Architectures", page_icon="🧭", layout="wide")
inject_css()
render_sidebar_api_key()

st.title("🧭 Modern RAG Architectures")
st.caption(
    "Six working Retrieval-Augmented Generation apps, each its own page, sharing one "
    "real retrieval stack: a FAISS vector index, BM25 keyword search, a networkx "
    "knowledge graph, live web search, and Claude for generation."
)

st.markdown(
    """
Each problem calls for a different retrieval approach. Pick a page in the sidebar
(or click a card below) to open that architecture's dedicated app: its own pipeline
diagram, when it works well vs. where it breaks, and a **live run** — real retrieval
against a real corpus, not a canned demo.
"""
)

st.markdown("---")

PAGE_FILES = {
    "classic": "pages/1_Classic_RAG.py",
    "hybrid": "pages/2_Hybrid_RAG.py",
    "graph": "pages/3_Graph_RAG.py",
    "agentic": "pages/4_Agentic_RAG.py",
    "corrective": "pages/5_Corrective_RAG.py",
    "multimodal": "pages/6_Multimodal_RAG.py",
}

cols = st.columns(3)
for i, spec in enumerate(ALL_SPECS):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="arch-card">
                <h3>{spec.icon} {spec.name}</h3>
                <p><em>{spec.tagline}</em></p>
                <p style="font-size:0.9rem;">{spec.description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link(PAGE_FILES[spec.key], label=f"Open {spec.name} →")
        st.markdown("")

st.markdown("---")
st.page_link("pages/7_Compare_All.py", label="📊 Open the side-by-side comparison →")

st.markdown("---")
st.markdown(
    '<div class="callout">💡 <strong>These patterns compose.</strong> An agentic '
    "workflow can use hybrid search, retrieve graph context, inspect a chart, and "
    "evaluate the evidence before answering. Every addition also introduces "
    "latency, cost, and more behavior to evaluate — pick the smallest architecture "
    "that solves the problem in front of you.</div>",
    unsafe_allow_html=True,
)
