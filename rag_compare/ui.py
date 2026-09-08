"""Reusable rendering helpers shared by every page.

Keeping this logic in one place means each architecture's page file is just
"load the spec, call render_architecture_page" — the pipeline diagram,
callout box, and try-it walkthrough all render the same way everywhere.
"""

from __future__ import annotations

import streamlit as st

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.llm import LlmConfig

_CSS = """
<style>
.pipeline-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.6rem 0;
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
.pipeline-down {
    font-size: 1.4rem;
    color: #9ca3af;
    font-weight: bold;
    margin: -0.2rem 0 0.2rem 0;
}
.callout {
    border-left: 4px solid #6366f1;
    background: rgba(99, 102, 241, 0.08);
    padding: 0.8rem 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
}
.quote {
    border-left: 4px solid #9ca3af;
    padding: 0.6rem 1rem;
    color: inherit;
    opacity: 0.9;
    margin: 0.8rem 0 1.2rem 0;
    font-style: italic;
}
.arch-card {
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    height: 100%;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _box(label: str, color: str) -> str:
    return f'<div class="pipeline-box" style="--box-bg:{color}">{label}</div>'


def render_pipeline(segments: list[PipelineSegment]) -> None:
    """Render a full multi-row pipeline diagram from architecture-neutral segments."""
    for i, segment in enumerate(segments):
        if i > 0:
            st.markdown('<div class="pipeline-down">&#8595;</div>', unsafe_allow_html=True)

        if segment.kind == "row":
            parts = ['<div class="pipeline-row">']
            stages = segment.stages
            for j, (label, color) in enumerate(stages):
                parts.append(_box(label, color))
                if j != len(stages) - 1:
                    parts.append('<div class="pipeline-arrow">&#8594;</div>')
            parts.append("</div>")
            st.markdown("".join(parts), unsafe_allow_html=True)
        elif segment.kind == "parallel":
            branches = segment.stages
            cols = st.columns(len(branches))
            for col, branch in zip(cols, branches):
                with col:
                    parts = ['<div class="pipeline-row">']
                    for label, color in branch:
                        parts.append(_box(label, color))
                    parts.append("</div>")
                    st.markdown("".join(parts), unsafe_allow_html=True)
        else:
            raise ValueError(f"Unknown pipeline segment kind: {segment.kind!r}")

        if segment.caption:
            st.caption(segment.caption)


def render_sidebar_llm_config() -> LlmConfig | None:
    """Shared LLM provider picker, persisted in session_state so it carries
    across every page in the multipage app. Retrieval is always real; this
    only controls whether generation uses a real LLM or the local
    extractive fallback — and if an LLM, which one: native Anthropic, or
    any OpenAI-compatible endpoint by URL, key, and model name."""
    st.sidebar.markdown("### 🔑 LLM provider (optional)")
    provider_label = st.sidebar.selectbox(
        "Provider",
        ["None (extractive fallback)", "Anthropic (Claude)", "Custom / OpenAI-compatible URL"],
        key="llm_provider_choice",
        help=(
            "Retrieval is always real regardless of this setting. Pick a provider "
            "to have it generate the final answer; otherwise a real extractive "
            "summarizer picks the best sentences from the retrieved context."
        ),
    )

    if provider_label.startswith("None"):
        st.sidebar.caption("No provider selected — using the extractive fallback for generation.")
        return None

    api_key = st.sidebar.text_input("API key", type="password", key="llm_api_key")

    if provider_label.startswith("Anthropic"):
        model = st.sidebar.text_input("Model", value="claude-sonnet-5", key="llm_model_anthropic")
        base_url = st.sidebar.text_input(
            "Base URL (optional override)",
            value="",
            key="llm_base_url_anthropic",
            help="Leave blank for api.anthropic.com, or point at a compatible proxy/gateway.",
        )
        if not api_key:
            st.sidebar.caption("Enter an API key to enable Claude generation.")
            return None
        st.sidebar.success(f"Using Anthropic ({model}) for generation.")
        return LlmConfig(provider="anthropic", api_key=api_key, base_url=base_url or None, model=model or None)

    base_url = st.sidebar.text_input(
        "Base URL",
        value="https://api.openai.com/v1",
        key="llm_base_url_custom",
        help="Any endpoint implementing POST {base_url}/chat/completions, e.g. OpenAI, "
        "Groq, Together, OpenRouter, or a local Ollama/vLLM server.",
    )
    model = st.sidebar.text_input("Model name", value="gpt-4o-mini", key="llm_model_custom")
    if not api_key or not base_url or not model:
        st.sidebar.caption("Enter a base URL, model name, and API key to enable generation.")
        return None
    st.sidebar.success(f"Using {base_url} ({model}) for generation.")
    return LlmConfig(provider="openai_compatible", api_key=api_key, base_url=base_url, model=model)


def works_well_breaks(good_points: list[str], bad_points: list[str]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Works well when**")
        for p in good_points:
            st.markdown(f"- {p}")
    with col2:
        st.markdown("**⚠️ Where it breaks**")
        for p in bad_points:
            st.markdown(f"- {p}")


def render_architecture_page(spec: ArchitectureSpec) -> None:
    """Render a complete, self-contained page for one architecture."""
    inject_css()
    llm_config = render_sidebar_llm_config()

    st.title(f"{spec.icon} {spec.name} — {spec.tagline}")
    st.markdown(f'<div class="callout">{spec.description}</div>', unsafe_allow_html=True)

    st.subheader("Pipeline")
    render_pipeline(spec.pipeline)

    st.markdown(f'<div class="quote">{spec.example_quote}</div>', unsafe_allow_html=True)

    works_well_breaks(spec.good_points, spec.bad_points)

    st.subheader("Try it — real retrieval, real generation")
    st.caption(
        "This runs the actual retrieval stack for this architecture (a FAISS "
        "vector index, BM25, a networkx knowledge graph, and/or live web search, "
        "depending on the page) against the demo corpus — nothing here is scripted."
    )
    query = st.text_input("Ask a question", value=spec.default_query, key=f"{spec.key}_query")

    kwargs = {}
    if spec.slider is not None:
        kwargs[spec.slider.key] = st.slider(
            spec.slider.label,
            spec.slider.min_value,
            spec.slider.max_value,
            spec.slider.default,
            key=f"{spec.key}_{spec.slider.key}",
        )

    if st.button(f"Run {spec.name}", key=f"{spec.key}_run"):
        with st.status(f"Running {spec.name}...", expanded=True) as status:
            result: SimResult = spec.simulate(query, llm_config=llm_config, **kwargs)
            for step in result.steps:
                st.write(step)
            status.update(label="Retrieval + generation complete", state="complete")

        st.subheader("Answer")
        st.success(result.answer)

        if result.image_bytes is not None:
            st.image(result.image_bytes, caption=result.image_caption)
        if result.dataframe is not None:
            st.dataframe(result.dataframe, use_container_width=True)
        if result.graph_image_bytes is not None:
            st.image(result.graph_image_bytes, caption="Traversed knowledge graph (matched entities highlighted)")
