"""Real web search via the DuckDuckGo Instant Answer API (no API key
required). Used by Agentic RAG (external source) and Corrective RAG
(supplementing weak retrieval). Network access is optional — if it's
unavailable (offline, sandboxed, proxy-restricted) this fails gracefully
and the caller falls back to local knowledge only."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import TypedDict


class WebResult(TypedDict):
    available: bool
    snippet: str
    source: str | None


def duckduckgo_search(query: str, timeout: float = 4.0) -> WebResult:
    try:
        url = (
            "https://api.duckduckgo.com/?q="
            + urllib.parse.quote(query)
            + "&format=json&no_html=1&skip_disambig=1"
        )
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
            data = json.load(resp)

        abstract = (data.get("AbstractText") or "").strip()
        if abstract:
            return {
                "available": True,
                "snippet": abstract,
                "source": data.get("AbstractURL") or "duckduckgo.com",
            }

        for related in data.get("RelatedTopics") or []:
            if isinstance(related, dict) and related.get("Text"):
                return {
                    "available": True,
                    "snippet": related["Text"],
                    "source": related.get("FirstURL") or "duckduckgo.com",
                }

        return {"available": False, "snippet": "No external result found.", "source": None}
    except Exception as exc:  # noqa: BLE001 - network can fail for many reasons
        return {
            "available": False,
            "snippet": f"Web search unavailable ({exc.__class__.__name__}: {exc}).",
            "source": None,
        }
