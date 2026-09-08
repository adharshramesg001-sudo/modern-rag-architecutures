"""Answer generation.

Generation is provider-agnostic: the caller supplies an `LlmConfig` (never
hardcoded — always entered in the sidebar) naming either "anthropic" or
any OpenAI-compatible REST endpoint via a base URL, API key, and model
name. That covers OpenAI, Groq, Together, Fireworks, DeepSeek, Mistral,
OpenRouter, a local Ollama/vLLM/LM Studio server, or any other provider
that speaks the `POST {base_url}/chat/completions` schema — no extra SDK
required, just `urllib`.

Without a config (or without a key), every architecture still works
end-to-end: answers fall back to a real (if simple) extractive summarizer
that scores each sentence in the retrieved context by term overlap with
the question and returns the best ones — not a canned string.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple, Union

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


@dataclass
class LlmConfig:
    """Where to send generation requests.

    provider: "anthropic" (native Anthropic Messages API) or
        "openai_compatible" (any REST endpoint implementing the OpenAI
        chat-completions schema).
    api_key: sent as the provider's bearer/auth token. Never persisted
        beyond the current session.
    base_url: required for "openai_compatible"; optional override for
        "anthropic" (e.g. a compatible proxy or gateway).
    model: model name/ID to request. Defaults to DEFAULT_ANTHROPIC_MODEL
        for "anthropic" if left blank; required for "openai_compatible".
    """

    provider: str
    api_key: str
    base_url: Optional[str] = None
    model: Optional[str] = None


def generate_answer(question: str, context: str, config: Optional[LlmConfig]) -> Tuple[str, bool]:
    """Returns (answer_text, used_llm)."""
    if not context.strip():
        return "No relevant context was retrieved to answer this question.", False

    if config and config.api_key:
        prompt = (
            "Answer the question using only the context below. Be concise "
            "(2-4 sentences). If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        try:
            return _call_chat(config, prompt), True
        except Exception as exc:  # noqa: BLE001 - many possible provider/network errors
            fallback = _extractive_summary(question, context)
            return f"[{config.provider} call failed, using extractive fallback: {exc}]\n\n{fallback}", False

    return _extractive_summary(question, context), False


def describe_image(question: str, image_bytes: bytes, config: Optional[LlmConfig]) -> Tuple[str, bool]:
    """Real vision call if a config+key is supplied; empty string (with
    used_llm=False) otherwise so the caller can fall back to a computed
    text description instead."""
    if not config or not config.api_key:
        return "", False
    try:
        return _call_chat(config, question, image_bytes=image_bytes), True
    except Exception as exc:  # noqa: BLE001
        return f"[Vision call failed: {exc}]", False


def _call_chat(config: LlmConfig, prompt: str, image_bytes: Optional[bytes] = None) -> str:
    if config.provider == "anthropic":
        return _call_anthropic(config, prompt, image_bytes)
    if config.provider == "openai_compatible":
        return _call_openai_compatible(config, prompt, image_bytes)
    raise ValueError(f"Unknown provider: {config.provider!r}")


def _call_anthropic(config: LlmConfig, prompt: str, image_bytes: Optional[bytes]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=config.api_key, base_url=config.base_url or None)
    content: Union[str, list] = prompt
    if image_bytes is not None:
        encoded = base64.b64encode(image_bytes).decode()
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}},
            {"type": "text", "text": prompt},
        ]
    response = client.messages.create(
        model=config.model or DEFAULT_ANTHROPIC_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text.strip()


def _call_openai_compatible(config: LlmConfig, prompt: str, image_bytes: Optional[bytes]) -> str:
    if not config.base_url:
        raise ValueError("A base URL is required for a custom OpenAI-compatible provider.")
    if not config.model:
        raise ValueError("A model name is required for a custom OpenAI-compatible provider.")

    content: Union[str, list] = prompt
    if image_bytes is not None:
        encoded = base64.b64encode(image_bytes).decode()
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]

    url = config.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": config.model,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - user-supplied API endpoint
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:300]}") from exc

    return data["choices"][0]["message"]["content"].strip()


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
