"""Hybrid RAG — Combine.

Runs real semantic search (FAISS/TF-IDF) and real keyword search (BM25) in
parallel, then merges both rankings with reciprocal rank fusion — so an
exact identifier like an error code is never lost to a purely semantic
embedding, and a paraphrase is never lost to a purely lexical match.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.corpus import DOCUMENTS
from rag_compare.llm import LlmConfig, generate_answer
from rag_compare.retrieval.fusion import reciprocal_rank_fusion
from rag_compare.retrieval.keyword_store import Bm25Store
from rag_compare.retrieval.vector_store import TfidfVectorStore

_vector_store = TfidfVectorStore(DOCUMENTS)
_keyword_store = Bm25Store(DOCUMENTS)


def simulate(query: str, llm_config: Optional[LlmConfig] = None) -> SimResult:
    semantic_hits = _vector_store.search(query, k=3)
    keyword_hits = _keyword_store.search(query, k=3)

    steps = [f"1. Query: \"{query}\"", "2a. Semantic search (FAISS/TF-IDF):"]
    for h in semantic_hits:
        steps.append(f"    • score={h['score']:.3f} — {h['doc']['title']}")
    steps.append("2b. Keyword search (BM25):")
    for h in keyword_hits:
        steps.append(f"    • score={h['score']:.3f} — {h['doc']['title']}")

    fused = reciprocal_rank_fusion([semantic_hits, keyword_hits], top_n=3)
    steps.append("3. Merge & re-rank both rankings with Reciprocal Rank Fusion:")
    for h in fused:
        steps.append(f"    • fused_score={h['score']:.4f} — {h['doc']['title']}")

    context = "\n\n".join(f"{h['doc']['title']}: {h['doc']['text']}" for h in fused)
    answer, used_llm = generate_answer(query, context, llm_config)
    generator = "LLM generated" if used_llm else "Extractive fallback synthesized"
    steps.append(f"4. {generator} the answer from the fused top results.")

    return SimResult(steps=steps, answer=answer)


SPEC = ArchitectureSpec(
    key="hybrid",
    name="Hybrid RAG",
    tagline="Combine",
    icon="🔀",
    description=(
        "When meaning alone is not enough. Combines semantic search with keyword "
        "retrieval, then merges the results."
    ),
    pipeline=[
        PipelineSegment("row", (("Query", "#2563eb"),)),
        PipelineSegment(
            "parallel",
            (
                (("Semantic Search<br>(FAISS / TF-IDF)", "#7c3aed"),),
                (("Keyword Search<br>(BM25)", "#b45309"),),
            ),
        ),
        PipelineSegment(
            "row",
            (
                ("Merge &<br>Re-rank (RRF)", "#334155"),
                ("LLM", "#0f766e"),
                ("Answer", "#059669"),
            ),
        ),
    ],
    example_quote=(
        "&#8220;Find the troubleshooting guide for error E104&#8221; needs both meaning "
        "<em>and</em> precision — a pure embedding search can blur past an exact "
        "identifier like <code>E104</code>, while keyword search alone misses "
        "paraphrased questions."
    ),
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
    default_query="Find the troubleshooting guide for error E104",
    simulate=simulate,
    compare_row={
        "Core idea": "Combine",
        "Retrieval hops": "Single-hop, dual-index",
        "Data source(s)": "Vector DB + Keyword index (BM25)",
        "Adapts mid-query": "No",
        "Self-correcting": "No",
        "Latency / cost": "Low-Medium",
        "Best for": "Queries mixing plain language with exact identifiers/codes",
        "Main weakness": "Two systems to tune and keep in sync",
    },
)
