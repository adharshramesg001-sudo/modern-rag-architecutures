"""Small, dependency-free scoring heuristics used for self-evaluation
(Agentic RAG) and relevance grading (Corrective RAG). Deterministic and
real: computed from the actual query and context text, not scripted."""

from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and",
    "for", "on", "with", "our", "latest", "what", "how", "which", "does",
    "do", "did", "at", "by", "this", "that", "it", "as", "be", "or",
}


def significant_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def keyword_coverage(query: str, context: str) -> float:
    """Fraction of the query's significant terms present in the context.
    Used as a cheap, real proxy for "does this context actually answer the
    question" — no LLM call required."""
    terms = significant_terms(query)
    if not terms:
        return 1.0
    ctx = context.lower()
    hits = sum(1 for term in terms if term in ctx)
    return hits / len(terms)
