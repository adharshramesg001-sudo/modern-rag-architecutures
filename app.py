import streamlit as st

st.set_page_config(
    page_title="Modern RAG Architectures",
    page_icon="🧭",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .pipeline-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 1.2rem 0 1.6rem 0;
    }
    .pipeline-box {
        background: var(--box-bg, #1f2937);
        color: white;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        font-weight: 600;
        font-size: 0.92rem;
        text-align: center;
        line-height: 1.25;
        min-width: 110px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }
    .pipeline-arrow {
        font-size: 1.4rem;
        color: #9ca3af;
        font-weight: bold;
    }
    .tag-good {
        background: #10b981;
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.4rem;
    }
    .tag-bad {
        background: #ef4444;
        color: white;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.4rem;
    }
    .callout {
        border-left: 4px solid #6366f1;
        background: rgba(99, 102, 241, 0.08);
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_pipeline(steps, colors=None):
    """Render a horizontal flow of boxes connected by arrows."""
    default_color = "#1f2937"
    parts = ['<div class="pipeline-row">']
    for i, step in enumerate(steps):
        color = colors[i] if colors and i < len(colors) else default_color
        parts.append(
            f'<div class="pipeline-box" style="--box-bg:{color}">{step}</div>'
        )
        if i != len(steps) - 1:
            parts.append('<div class="pipeline-arrow">&#8594;</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def works_well_breaks(good_points, bad_points, good_label="Works well when", bad_label="Where it breaks"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**✅ {good_label}**")
        for p in good_points:
            st.markdown(f"- {p}")
    with col2:
        st.markdown(f"**⚠️ {bad_label}**")
        for p in bad_points:
            st.markdown(f"- {p}")


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧭 Modern RAG Architectures")
st.caption(
    "Three ways to build Retrieval-Augmented Generation systems — from a single "
    "similarity search to a self-correcting reasoning agent."
)

(
    tab_classic,
    tab_hybrid,
    tab_graph,
    tab_agentic,
    tab_corrective,
    tab_multimodal,
    tab_compare,
) = st.tabs(
    [
        "🔹 Classic RAG",
        "🔀 Hybrid RAG",
        "🕸️ Graph RAG",
        "🤖 Agentic RAG",
        "🛠️ Corrective RAG",
        "🖼️ Multimodal RAG",
        "📊 Compare All",
    ]
)

# ----------------------------------------------------------------------------
# Classic RAG
# ----------------------------------------------------------------------------
with tab_classic:
    st.header("Classic RAG — Retrieve")
    st.markdown(
        '<div class="callout">Simple, fast, single-hop. Embed the query, '
        "find the closest chunks, hand them to the LLM.</div>",
        unsafe_allow_html=True,
    )

    render_pipeline(
        ["Query", "Embed", "Vector DB", "Top-K<br>Chunks", "LLM", "Answer"],
        colors=["#2563eb", "#2563eb", "#7c3aed", "#7c3aed", "#0f766e", "#059669"],
    )

    works_well_breaks(
        good_points=[
            "You're searching a document set and semantic similarity alone gets you to the right chunk",
            "Low-latency, single-turn Q&A over a relatively self-contained corpus",
            "Simplicity and predictable cost matter more than nuance",
        ],
        bad_points=[
            "No relationship traversal — can't connect facts across documents",
            "No adapting mid-query — if the first retrieval misses, there's no recovery",
            "One shot and it's done — no iteration, no self-checking",
        ],
    )

    st.subheader("Try it (simulated)")
    query = st.text_input(
        "Ask a question", value="What is the refund policy?", key="classic_query"
    )
    if st.button("Run Classic RAG", key="classic_run"):
        steps = [
            f"1. Embed query: *\"{query}\"*",
            "2. Search vector DB for nearest neighbor chunks",
            "3. Retrieve Top-K chunks (single pass, no re-ranking loop)",
            "4. Stuff chunks into LLM prompt",
            "5. Return answer",
        ]
        with st.status("Running single-hop retrieval...", expanded=True) as status:
            for s in steps:
                st.write(s)
            status.update(label="Done — one retrieval pass, no looping back", state="complete")

# ----------------------------------------------------------------------------
# Hybrid RAG
# ----------------------------------------------------------------------------
with tab_hybrid:
    st.header("Hybrid RAG — Combine")
    st.markdown(
        '<div class="callout">When meaning alone is not enough. Combines semantic '
        "search with keyword retrieval, then merges the results.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Pipeline**")
    render_pipeline(
        ["Query"],
        colors=["#2563eb"],
    )
    col_a, col_b = st.columns(2)
    with col_a:
        render_pipeline(["Semantic<br>Search"], colors=["#7c3aed"])
    with col_b:
        render_pipeline(["Keyword<br>Search (BM25)"], colors=["#b45309"])
    render_pipeline(
        ["Merge &<br>Re-rank", "LLM", "Answer"],
        colors=["#334155", "#0f766e", "#059669"],
    )

    st.markdown(
        "> *When meaning alone is not enough:* \"Find the troubleshooting guide for "
        "error E104\" needs both meaning **and** precision — a pure embedding search can "
        "blur past an exact identifier like `E104`, while keyword search alone misses "
        "paraphrased questions."
    )

    works_well_breaks(
        good_points=[
            "Users ask questions in plain language that also contain exact identifiers, technical terms, or error codes",
            "Corpus mixes free-text prose with structured/precise tokens (SKUs, codes, IDs, acronyms)",
            "You want the recall of embeddings with the precision of exact-match search",
        ],
        bad_points=[
            "Two retrieval systems to run, tune, and keep in sync (extra infra)",
            "Merging/re-ranking logic adds a tuning surface (how to weigh semantic vs. keyword scores)",
            "Overkill when queries are purely conversational with no exact-match terms",
        ],
    )

    st.subheader("Try it (simulated)")
    query_h = st.text_input(
        "Ask a question", value="Find the troubleshooting guide for error E104", key="hybrid_query"
    )
    if st.button("Run Hybrid RAG", key="hybrid_run"):
        with st.status("Running hybrid retrieval...", expanded=True) as status:
            st.write(f"1. Query: *\"{query_h}\"*")
            st.write("2a. Semantic search → chunks about troubleshooting/error guides")
            st.write("2b. Keyword search (BM25) → exact matches on `E104`")
            st.write("3. Merge both result sets and re-rank by combined score")
            st.write("4. Pass top merged results to LLM")
            st.write("5. Return answer")
            status.update(label="Done — precision of keyword search + recall of semantic search", state="complete")

# ----------------------------------------------------------------------------
# Graph RAG
# ----------------------------------------------------------------------------
with tab_graph:
    st.header("Graph RAG — Connect")
    st.markdown(
        '<div class="callout">Relationship-driven, entity-heavy. Instead of pulling '
        "similar-looking chunks, it identifies entities in the query, walks the graph "
        "connecting them, and builds context around those relationships.</div>",
        unsafe_allow_html=True,
    )

    render_pipeline(
        ["Query", "Entity<br>Extraction", "Knowledge<br>Graph", "Connected<br>Context", "LLM", "Answer"],
        colors=["#2563eb", "#b45309", "#b91c1c", "#b91c1c", "#0f766e", "#059669"],
    )

    st.markdown(
        "> *When the answer depends on connections:* \"Which projects depend on "
        "suppliers affected by this incident?\" — the evidence lives across connected facts, "
        "not inside any single chunk."
    )

    works_well_breaks(
        good_points=[
            "The answer isn't in one document — it's in how entities link to each other across your data",
            "Multi-hop questions (\"how is X related to Y through Z?\")",
            "Domains with rich, explicit relationships: orgs, people, products, supply chains",
        ],
        bad_points=[
            "Requires building and maintaining a knowledge graph up front — real cost and effort",
            "Entity extraction quality caps answer quality (missed/mislabeled entities = missed links)",
            "Overkill for simple, single-document lookups that plain vector search already solves",
        ],
    )

    st.subheader("Try it (simulated)")
    query_g = st.text_input(
        "Ask a question", value="How is Acme Corp connected to the FDA recall?", key="graph_query"
    )
    if st.button("Run Graph RAG", key="graph_run"):
        steps = [
            f"1. Extract entities from: *\"{query_g}\"* → `Acme Corp`, `FDA recall`",
            "2. Locate these entities as nodes in the knowledge graph",
            "3. Walk edges connecting them (e.g. Acme Corp → Product X → Recall Notice → FDA)",
            "4. Build context from the connected subgraph (multiple sources, not one chunk)",
            "5. Pass entity-linked context to LLM",
            "6. Return answer grounded in the relationship path",
        ]
        with st.status("Traversing knowledge graph...", expanded=True) as status:
            for s in steps:
                st.write(s)
            status.update(label="Done — answer built from relationship traversal, not just similarity", state="complete")

# ----------------------------------------------------------------------------
# Agentic RAG
# ----------------------------------------------------------------------------
with tab_agentic:
    st.header("Agentic RAG — Reason")
    st.markdown(
        '<div class="callout">Adaptive, iterative, self-correcting. The agent itself '
        "decides what to retrieve, where to look, and whether to loop back for another pass.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Pipeline**")
    render_pipeline(
        ["Query", "Reasoning<br>Agent"],
        colors=["#2563eb", "#7c3aed"],
    )
    st.markdown(
        '<div class="pipeline-row" style="margin-top:-1rem;">'
        '<div class="pipeline-arrow">&#8595;</div></div>',
        unsafe_allow_html=True,
    )
    render_pipeline(
        ["Vector DB", "Knowledge<br>Graph", "Web Search", "Tools"],
        colors=["#0f766e", "#b91c1c", "#b45309", "#334155"],
    )
    st.caption("↑ the agent chooses one, several, or repeated rounds of the above ↑")
    render_pipeline(
        ["Self-Evaluation", "Final Answer"],
        colors=["#7c3aed", "#059669"],
    )

    st.markdown(
        "> *When retrieval requires several decisions:* one question might require "
        "vector search, SQL, **and** a web lookup. Give that loop clear stopping "
        "conditions and a budget — otherwise it will keep looping."
    )

    works_well_breaks(
        good_points=[
            "The query is complex enough to need multiple retrieval rounds",
            "Source-switching is required (vector DB isn't enough — needs SQL, web, or other tools too)",
            "A quality check before the answer ships matters (high-stakes or ambiguous asks)",
        ],
        bad_points=[
            "Higher latency and cost — multiple LLM calls and tool round-trips per query",
            "Harder to debug/predict — the agent's path isn't fixed ahead of time",
            "Without clear stopping conditions and a budget, the loop can run away",
        ],
    )

    st.subheader("Try it (simulated)")
    query_a = st.text_input(
        "Ask a question",
        value="Compare our Q3 revenue trend to the latest industry benchmark",
        key="agentic_query",
    )
    max_loops = st.slider("Max reasoning loops", 1, 4, 2, key="agentic_loops")
    if st.button("Run Agentic RAG", key="agentic_run"):
        with st.status("Agent reasoning...", expanded=True) as status:
            st.write(f"1. Receive query: *\"{query_a}\"*")
            st.write("2. Agent plans: needs internal financials + external benchmark data")
            for i in range(1, max_loops + 1):
                st.write(f"**Round {i}**")
                if i == 1:
                    st.write("   • Retrieve Q3 figures from Vector DB")
                    st.write("   • Retrieve org relationships from Knowledge Graph")
                elif i == 2:
                    st.write("   • Web Search for latest industry benchmark report")
                else:
                    st.write("   • Follow-up tool call to fill remaining gaps")
                st.write("   • Self-evaluate: is the context sufficient and consistent?")
            st.write("3. Self-evaluation passes confidence threshold")
            st.write("4. Synthesize final answer from all gathered sources")
            status.update(label=f"Done — converged after {max_loops} reasoning round(s)", state="complete")

# ----------------------------------------------------------------------------
# Corrective RAG
# ----------------------------------------------------------------------------
with tab_corrective:
    st.header("Corrective RAG — Check")
    st.markdown(
        '<div class="callout">When retrieved evidence needs checking. Evaluates '
        "retrieved material, then refines or supplements it before generation — "
        "retrieval returning results is only the beginning.</div>",
        unsafe_allow_html=True,
    )

    render_pipeline(
        ["Query", "Retrieve", "Evaluate<br>Relevance", "Refine /<br>Supplement", "LLM", "Answer"],
        colors=["#2563eb", "#7c3aed", "#b45309", "#b91c1c", "#0f766e", "#059669"],
    )
    st.caption(
        "If evaluation finds the retrieved chunks weak or off-target, the loop back to "
        "\"Refine / Supplement\" can re-query the vector DB, fall back to web search, "
        "or strip out irrelevant chunks before generation."
    )

    st.markdown(
        "> *When retrieved evidence needs checking:* those results still need to "
        "support the question — a relevance grader sits between retrieval and generation "
        "instead of trusting Top-K blindly."
    )

    works_well_breaks(
        good_points=[
            "Retrieval quality is inconsistent or the corpus is noisy",
            "Wrong or irrelevant context is costly (compliance, medical, legal, financial answers)",
            "You need a safety net that catches bad retrieval before it reaches the LLM",
        ],
        bad_points=[
            "Extra evaluation step adds latency and an additional LLM/grader call",
            "Grader quality becomes a new failure point — a bad grader can discard good context",
            "Unnecessary overhead when the corpus is small, clean, and retrieval is already reliable",
        ],
    )

    st.subheader("Try it (simulated)")
    query_c = st.text_input(
        "Ask a question", value="What are the side effects of drug X at dose Y?", key="corrective_query"
    )
    if st.button("Run Corrective RAG", key="corrective_run"):
        with st.status("Retrieving and checking evidence...", expanded=True) as status:
            st.write(f"1. Query: *\"{query_c}\"*")
            st.write("2. Retrieve Top-K chunks from vector DB")
            st.write("3. Evaluate relevance of each chunk against the query")
            st.write("   • 2 chunks judged strong, 1 chunk judged weak/off-topic")
            st.write("4. Refine: drop the weak chunk, supplement with a fresh web search")
            st.write("5. Pass corrected, supported context to LLM")
            st.write("6. Return answer")
            status.update(label="Done — answer grounded only in evidence that passed the check", state="complete")

# ----------------------------------------------------------------------------
# Multimodal RAG
# ----------------------------------------------------------------------------
with tab_multimodal:
    st.header("Multimodal RAG — Expand")
    st.markdown(
        '<div class="callout">When knowledge extends beyond text. Retrieves evidence '
        "from text, images, charts, and tables for a model that can interpret the "
        "relevant content.</div>",
        unsafe_allow_html=True,
    )

    render_pipeline(
        ["Query", "Multimodal<br>Retrieval", "Text +<br>Images/Charts/<br>Tables", "Vision-Capable<br>LLM", "Answer"],
        colors=["#2563eb", "#7c3aed", "#334155", "#0f766e", "#059669"],
    )

    st.markdown(
        "> *When knowledge extends beyond text:* think reports where the decisive "
        "information sits inside a figure — a bar chart, a table of financials, a "
        "diagram — not the surrounding prose."
    )

    works_well_breaks(
        good_points=[
            "Source documents are PDFs/reports where key facts live in charts, tables, or images",
            "Text-only retrieval would miss the answer because it's visual, not written",
            "You have a vision-capable model that can reason over retrieved images directly",
        ],
        bad_points=[
            "Indexing images/charts/tables is harder than indexing text (layout parsing, OCR, chart understanding)",
            "Retrieval across modalities needs either multimodal embeddings or separate per-modality indexes",
            "More moving parts and higher inference cost than text-only retrieval",
        ],
    )

    st.subheader("Try it (simulated)")
    query_m = st.text_input(
        "Ask a question",
        value="What was the revenue trend shown in the Q3 earnings chart?",
        key="multimodal_query",
    )
    if st.button("Run Multimodal RAG", key="multimodal_run"):
        with st.status("Retrieving across modalities...", expanded=True) as status:
            st.write(f"1. Query: *\"{query_m}\"*")
            st.write("2. Multimodal retrieval scans text, chart images, and tables")
            st.write("3. Top match: a chart image on page 12 of the Q3 earnings report")
            st.write("4. Pass the chart image + surrounding table/text to a vision-capable LLM")
            st.write("5. Model interprets the chart directly")
            st.write("6. Return answer grounded in the figure, not just the caption")
            status.update(label="Done — answer sourced from a chart, not just surrounding text", state="complete")

# ----------------------------------------------------------------------------
# Compare
# ----------------------------------------------------------------------------
with tab_compare:
    st.header("Side-by-Side Comparison")

    import pandas as pd

    data = {
        "Dimension": [
            "Core idea",
            "Retrieval hops",
            "Data source(s)",
            "Adapts mid-query",
            "Self-correcting",
            "Latency / cost",
            "Best for",
            "Main weakness",
        ],
        "Classic RAG": [
            "Retrieve",
            "Single-hop",
            "Vector DB",
            "No",
            "No",
            "Low",
            "Semantic lookup in a document set",
            "No relationship traversal, one shot only",
        ],
        "Hybrid RAG": [
            "Combine",
            "Single-hop, dual-index",
            "Vector DB + Keyword index (BM25)",
            "No",
            "No",
            "Low-Medium",
            "Queries mixing plain language with exact identifiers/codes",
            "Two systems to tune and keep in sync",
        ],
        "Graph RAG": [
            "Connect",
            "Multi-hop via graph edges",
            "Knowledge Graph (+ sources it indexes)",
            "No",
            "No",
            "Medium",
            "Answers that depend on how entities relate",
            "Needs a maintained knowledge graph",
        ],
        "Agentic RAG": [
            "Reason",
            "Iterative, agent-decided",
            "Vector DB + Knowledge Graph + Web + Tools (incl. SQL)",
            "Yes",
            "Yes",
            "High",
            "Complex, multi-step, high-stakes queries",
            "Slowest, most expensive; needs stopping conditions & budget",
        ],
        "Corrective RAG": [
            "Check",
            "Single-hop + evaluation loop",
            "Vector DB (+ web fallback)",
            "Partially (on evaluation failure)",
            "Yes",
            "Medium",
            "Cases where wrong context is costly",
            "Grader adds latency and is a new failure point",
        ],
        "Multimodal RAG": [
            "Expand",
            "Single-hop, cross-modality",
            "Text + Images + Charts + Tables",
            "No",
            "No",
            "Medium-High",
            "Reports where key facts live in figures, not text",
            "Harder to index; needs vision-capable models",
        ],
    }
    df = pd.DataFrame(data).set_index("Dimension")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Pipelines at a glance")

    st.markdown("**Classic RAG**")
    render_pipeline(
        ["Query", "Embed", "Vector DB", "Top-K", "LLM", "Answer"],
        colors=["#2563eb", "#2563eb", "#7c3aed", "#7c3aed", "#0f766e", "#059669"],
    )

    st.markdown("**Hybrid RAG**")
    render_pipeline(
        ["Query", "Semantic +<br>Keyword Search", "Merge &<br>Re-rank", "LLM", "Answer"],
        colors=["#2563eb", "#7c3aed", "#334155", "#0f766e", "#059669"],
    )

    st.markdown("**Graph RAG**")
    render_pipeline(
        ["Query", "Entity<br>Extraction", "Knowledge<br>Graph", "Connected<br>Context", "LLM", "Answer"],
        colors=["#2563eb", "#b45309", "#b91c1c", "#b91c1c", "#0f766e", "#059669"],
    )

    st.markdown("**Agentic RAG**")
    render_pipeline(
        ["Query", "Reasoning Agent", "(Vector DB + KG +<br>Web + Tools) ⟲", "Self-Eval", "Answer"],
        colors=["#2563eb", "#7c3aed", "#334155", "#7c3aed", "#059669"],
    )

    st.markdown("**Corrective RAG**")
    render_pipeline(
        ["Query", "Retrieve", "Evaluate<br>Relevance", "Refine /<br>Supplement", "LLM", "Answer"],
        colors=["#2563eb", "#7c3aed", "#b45309", "#b91c1c", "#0f766e", "#059669"],
    )

    st.markdown("**Multimodal RAG**")
    render_pipeline(
        ["Query", "Multimodal<br>Retrieval", "Text/Images/<br>Charts/Tables", "Vision LLM", "Answer"],
        colors=["#2563eb", "#7c3aed", "#334155", "#0f766e", "#059669"],
    )

    st.markdown("---")
    st.markdown(
        '<div class="callout">💡 <strong>These patterns compose.</strong> An agentic '
        "workflow can use hybrid search, retrieve graph context, inspect a chart, and "
        "evaluate the evidence before answering. Every addition also introduces "
        "latency, cost, and more behavior to evaluate — pick the smallest architecture "
        "that solves the problem in front of you.</div>",
        unsafe_allow_html=True,
    )

st.sidebar.title("About")
st.sidebar.write(
    "This app compares six Retrieval-Augmented Generation (RAG) architectures: "
    "**Classic**, **Hybrid**, **Graph**, **Agentic**, **Corrective**, and "
    "**Multimodal** RAG. Each tab breaks down its pipeline, when it works well, "
    "where it breaks, and a simulated walkthrough."
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit.")
