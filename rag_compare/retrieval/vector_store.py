"""A real vector database: FAISS for approximate/exact nearest-neighbor
search over TF-IDF embeddings computed locally with scikit-learn.

TF-IDF is used instead of a neural embedding model so the whole app runs
fully offline with no model download and no API key — but the vector
store itself, the index, and the similarity search are all real: FAISS
does an actual inner-product search over actual vectors built from the
actual corpus text.
"""

from __future__ import annotations

from typing import List, TypedDict

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag_compare.corpus import Document


class SearchHit(TypedDict):
    doc: Document
    score: float


class TfidfVectorStore:
    """Embeds a document set with TF-IDF and indexes it in FAISS for cosine search."""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        matrix = self.vectorizer.fit_transform([d["text"] for d in documents])
        vectors = matrix.toarray().astype("float32")
        faiss.normalize_L2(vectors)  # so inner product == cosine similarity
        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

    def search(self, query: str, k: int = 3) -> List[SearchHit]:
        if not query.strip():
            return []
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        faiss.normalize_L2(query_vec)
        k = min(k, len(self.documents))
        scores, indices = self.index.search(query_vec, k)
        hits: List[SearchHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            hits.append({"doc": self.documents[idx], "score": float(score)})
        return hits
