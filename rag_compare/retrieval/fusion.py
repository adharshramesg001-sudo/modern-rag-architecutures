"""Reciprocal Rank Fusion — merges multiple ranked result lists (e.g. a
semantic ranking and a keyword ranking) into one ranking, the standard
technique behind Hybrid RAG's "merge & re-rank" step."""

from __future__ import annotations

from typing import List

from rag_compare.retrieval.vector_store import SearchHit


def reciprocal_rank_fusion(
    result_lists: List[List[SearchHit]], k: int = 60, top_n: int = 5
) -> List[SearchHit]:
    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for results in result_lists:
        for rank, hit in enumerate(results):
            doc_id = hit["doc"]["id"]
            doc_lookup[doc_id] = hit["doc"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_n]
    return [{"doc": doc_lookup[doc_id], "score": fused_scores[doc_id]} for doc_id in ranked_ids]
