"""Classic RAG — Retrieve.

Single-hop: embed the query, search a real FAISS vector index built over
TF-IDF embeddings of the corpus, hand the top-k chunks to the answerer.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.corpus import DOCUMENTS
from rag_compare.llm import LlmConfig, generate_answer, model_label
from rag_compare.retrieval.vector_store import TfidfVectorStore
from rag_compare.tracing import Tracer, TracingConfig

_store = TfidfVectorStore(DOCUMENTS)


def simulate(
    query: str,
    llm_config: Optional[LlmConfig] = None,
    tracing_config: Optional[TracingConfig] = None,
) -> SimResult:
    tracer = Tracer(tracing_config, name="Classic RAG run", query=query)

    steps = [
        f"1. Vectorize the query with the TF-IDF model fit on all {len(DOCUMENTS)} corpus documents.",
        "2. Search the FAISS `IndexFlatIP` vector store for the nearest chunks by cosine similarity.",
    ]

    search_span = tracer.step("vector_search", as_type="retriever", input=query)
    hits = _store.search(query, k=3)
    search_span.update(output=[{"title": h["doc"]["title"], "score": h["score"]} for h in hits])
    search_span.end()

    if not hits:
        steps.append("   • No hits — empty query.")
    for hit in hits:
        preview = hit["doc"]["text"][:140].rstrip() + "..."
        steps.append(f"   • score={hit['score']:.3f} — **{hit['doc']['title']}**: {preview}")

    context = "\n\n".join(f"{h['doc']['title']}: {h['doc']['text']}" for h in hits)
    gen_span = tracer.step(
        "generate_answer", as_type="generation", input=context, model=model_label(llm_config)
    )
    answer, used_llm = generate_answer(query, context, llm_config)
    gen_span.update(output=answer)
    gen_span.end()

    generator = "LLM generated" if used_llm else "Extractive fallback synthesized"
    steps.append(
        f"3. {generator} the final answer from the retrieved chunks (single pass, no re-querying)."
    )

    trace_url = tracer.finish(output=answer)
    return SimResult(steps=steps, answer=answer, trace_url=trace_url)


SPEC = ArchitectureSpec(
    key="classic",
    name="Classic RAG",
    tagline="Retrieve",
    icon="🔹",
    description=(
        "Simple, fast, single-hop. Embed the query, find the closest chunks in a "
        "vector database, hand them to the LLM."
    ),
    pipeline=[
        PipelineSegment(
            "row",
            (
                ("Query", "#2563eb"),
                ("Embed", "#2563eb"),
                ("Vector DB", "#7c3aed"),
                ("Top-K<br>Chunks", "#7c3aed"),
                ("LLM", "#0f766e"),
                ("Answer", "#059669"),
            ),
        )
    ],
    example_quote=(
        "&#8220;What is the refund policy?&#8221; — the answer sits inside a single "
        "passage, so nearest-neighbor search over the vector DB finds it directly."
    ),
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
    default_query="What is the refund policy?",
    simulate=simulate,
    compare_row={
        "Core idea": "Retrieve",
        "Retrieval hops": "Single-hop",
        "Data source(s)": "Vector DB",
        "Adapts mid-query": "No",
        "Self-correcting": "No",
        "Latency / cost": "Low",
        "Best for": "Semantic lookup in a document set",
        "Main weakness": "No relationship traversal, one shot only",
    },
)
