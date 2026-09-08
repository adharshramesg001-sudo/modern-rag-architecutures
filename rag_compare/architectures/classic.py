"""Classic RAG — Retrieve.

Single-hop: embed the query, search a real FAISS vector index built over
TF-IDF embeddings of the corpus, hand the top-k chunks to the answerer.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.corpus import DOCUMENTS
from rag_compare.llm import generate_answer
from rag_compare.retrieval.vector_store import TfidfVectorStore

_store = TfidfVectorStore(DOCUMENTS)


def simulate(query: str, api_key: Optional[str] = None) -> SimResult:
    steps = [
        f"1. Vectorize the query with the TF-IDF model fit on all {len(DOCUMENTS)} corpus documents.",
        "2. Search the FAISS `IndexFlatIP` vector store for the nearest chunks by cosine similarity.",
    ]

    hits = _store.search(query, k=3)
    if not hits:
        steps.append("   • No hits — empty query.")
    for hit in hits:
        preview = hit["doc"]["text"][:140].rstrip() + "..."
        steps.append(f"   • score={hit['score']:.3f} — **{hit['doc']['title']}**: {preview}")

    context = "\n\n".join(f"{h['doc']['title']}: {h['doc']['text']}" for h in hits)
    answer, used_llm = generate_answer(query, context, api_key)
    generator = "Claude generated" if used_llm else "Extractive fallback synthesized"
    steps.append(
        f"3. {generator} the final answer from the retrieved chunks (single pass, no re-querying)."
    )

    return SimResult(steps=steps, answer=answer)


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
