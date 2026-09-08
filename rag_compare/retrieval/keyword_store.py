"""Real lexical/keyword search using BM25 (Okapi) — good at exact-match
terms like error codes or identifiers that a semantic embedding can blur."""

from __future__ import annotations

from typing import List

from rank_bm25 import BM25Okapi

from rag_compare.corpus import Document
from rag_compare.retrieval.vector_store import SearchHit


class Bm25Store:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        tokenized = [_tokenize(d["text"]) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 3) -> List[SearchHit]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: List[SearchHit] = []
        for idx in ranked[:k]:
            if scores[idx] <= 0:
                continue
            hits.append({"doc": self.documents[idx], "score": float(scores[idx])})
        return hits


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("/", " ").split()
