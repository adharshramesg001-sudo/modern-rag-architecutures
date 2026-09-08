"""Answer generation.

If the caller supplies an Anthropic API key (via the sidebar, never
hardcoded), the app makes a real call to Claude to generate the answer —
and, for Multimodal RAG, a real vision call to actually read a chart
image. Without a key, every architecture still works end-to-end: answers
fall back to a real (if simple) extractive summarizer that scores each
sentence in the retrieved context by term overlap with the question and
returns the best ones — not a canned string.
"""

from __future__ import annotations

import base64
import re
from typing import Optional, Tuple

MODEL = "claude-sonnet-5"


def generate_answer(question: str, context: str, api_key: Optional[str]) -> Tuple[str, bool]:
    """Returns (answer_text, used_llm)."""
    if not context.strip():
        return "No relevant context was retrieved to answer this question.", False

    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=MODEL,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question using only the context below. "
                            "Be concise (2-4 sentences). If the context doesn't "
                            "contain the answer, say so.\n\n"
                            f"Context:\n{context}\n\nQuestion: {question}"
                        ),
                    }
                ],
            )
            return response.content[0].text.strip(), True
        except Exception as exc:  # noqa: BLE001
            fallback = _extractive_summary(question, context)
            return f"[Claude call failed, using extractive fallback: {exc}]\n\n{fallback}", False

    return _extractive_summary(question, context), False


def describe_image(question: str, image_bytes: bytes, api_key: Optional[str]) -> Tuple[str, bool]:
    """Real vision call to Claude if a key is supplied; empty string (with
    used_llm=False) otherwise so the caller can fall back to a computed
    text description instead."""
    if not api_key:
        return "", False
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        encoded = base64.b64encode(image_bytes).decode()
        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": encoded},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        return response.content[0].text.strip(), True
    except Exception as exc:  # noqa: BLE001
        return f"[Vision call failed: {exc}]", False


def _extractive_summary(question: str, context: str, max_sentences: int = 3) -> str:
    """Score each sentence in the context by how many significant question
    terms it contains, and return the top-scoring sentences in their
    original order. A real, deterministic extractive answer."""
    from rag_compare.scoring import significant_terms  # local import: keep module import-light

    terms = significant_terms(question)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
    if not sentences:
        return context.strip()

    scored = [
        (i, sentence, sum(1 for term in terms if term in sentence.lower()))
        for i, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda t: t[2], reverse=True)
    top = sorted(scored[:max_sentences], key=lambda t: t[0])
    if all(score == 0 for _, _, score in top):
        top = [(0, sentences[0], 0)]
    return " ".join(sentence for _, sentence, _ in top)
