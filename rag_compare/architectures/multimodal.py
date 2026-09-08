"""Multimodal RAG — Expand.

Indexes real text, a real matplotlib-rendered chart (via its data-derived
caption), and real table rows in one retrieval pass, then hands whichever
modality actually matched to the answerer — including, when an API key is
supplied, a real Claude vision call that reads the chart image directly
instead of trusting a caption.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.corpus import DOCUMENTS, REVENUE_QUARTERS, REVENUE_VALUES_M
from rag_compare.llm import describe_image, generate_answer
from rag_compare.multimodal_data import revenue_chart_png, revenue_table, revenue_trend_description
from rag_compare.retrieval.vector_store import TfidfVectorStore

_TEXT_IDS = {"finance-q3", "multimodal-revenue-context"}

_index_items = [d for d in DOCUMENTS if d["id"] in _TEXT_IDS]
_index_items.append(
    {"id": "chart-revenue", "title": "Quarterly Revenue Chart", "text": revenue_trend_description()}
)
for quarter, value in zip(REVENUE_QUARTERS, REVENUE_VALUES_M):
    _index_items.append(
        {
            "id": f"table-row-{quarter}",
            "title": f"Revenue table row: {quarter}",
            "text": f"{quarter} revenue was ${value:.1f} million.",
        }
    )

_MODALITY = {
    "chart-revenue": "chart",
    **{f"table-row-{q}": "table" for q in REVENUE_QUARTERS},
    "finance-q3": "text",
    "multimodal-revenue-context": "text",
}

_store = TfidfVectorStore(_index_items)


def simulate(query: str, api_key: Optional[str] = None) -> SimResult:
    steps = [f"1. Query: \"{query}\"", "2. Multimodal retrieval scans text, chart captions, and table rows:"]

    hits = _store.search(query, k=3)
    for h in hits:
        modality = _MODALITY.get(h["doc"]["id"], "text")
        steps.append(f"    • [{modality}] score={h['score']:.3f} — {h['doc']['title']}")

    top_modality = _MODALITY.get(hits[0]["doc"]["id"], "text") if hits else "text"

    image_bytes = None
    dataframe = None

    if top_modality == "chart":
        steps.append("3. Top match is the chart — rendering the actual figure from the underlying data.")
        image_bytes = revenue_chart_png()
        vision_answer, used_vision = describe_image(query, image_bytes, api_key)
        if used_vision:
            steps.append("4. Claude vision call reads the chart image directly.")
            answer = vision_answer
        else:
            steps.append("4. No API key — falling back to the data-derived trend description.")
            context = revenue_trend_description()
            answer, _ = generate_answer(query, context, api_key)
    elif top_modality == "table":
        steps.append("3. Top match is a table row — attaching the underlying table.")
        dataframe = revenue_table()
        context = "\n".join(h["doc"]["text"] for h in hits)
        answer, used_llm = generate_answer(query, context, api_key)
        generator = "Claude generated" if used_llm else "Extractive fallback synthesized"
        steps.append(f"4. {generator} the answer from the matched table row(s).")
    else:
        context = "\n\n".join(f"{h['doc']['title']}: {h['doc']['text']}" for h in hits)
        answer, used_llm = generate_answer(query, context, api_key)
        generator = "Claude generated" if used_llm else "Extractive fallback synthesized"
        steps.append(f"3. {generator} the answer from the matched text passage(s).")

    return SimResult(
        steps=steps,
        answer=answer,
        image_bytes=image_bytes,
        image_caption="Quarterly Revenue — 2025 (rendered from the underlying data)",
        dataframe=dataframe,
    )


SPEC = ArchitectureSpec(
    key="multimodal",
    name="Multimodal RAG",
    tagline="Expand",
    icon="🖼️",
    description=(
        "When knowledge extends beyond text. Retrieves evidence from text, images, "
        "charts, and tables for a model that can interpret the relevant content."
    ),
    pipeline=[
        PipelineSegment(
            "row",
            (
                ("Query", "#2563eb"),
                ("Multimodal<br>Retrieval", "#7c3aed"),
                ("Text /<br>Chart /<br>Table", "#334155"),
                ("Vision-Capable<br>LLM", "#0f766e"),
                ("Answer", "#059669"),
            ),
        )
    ],
    example_quote=(
        "Think reports where the decisive information sits inside a figure — a bar "
        "chart, a table of financials, a diagram — not the surrounding prose."
    ),
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
    default_query="What was the revenue trend shown in the Q3 earnings chart?",
    simulate=simulate,
    compare_row={
        "Core idea": "Expand",
        "Retrieval hops": "Single-hop, cross-modality",
        "Data source(s)": "Text + Images + Charts + Tables",
        "Adapts mid-query": "No",
        "Self-correcting": "No",
        "Latency / cost": "Medium-High",
        "Best for": "Reports where key facts live in figures, not text",
        "Main weakness": "Harder to index; needs vision-capable models",
    },
)
