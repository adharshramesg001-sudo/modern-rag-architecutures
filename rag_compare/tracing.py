"""Optional Langfuse tracing.

If configured (public key, secret key, and host — entered in the sidebar,
never stored), each "Run" wraps its real retrieval + generation steps in
a Langfuse trace: one span per retrieval call (vector search, BM25, graph
traversal, web search, relevance grading, self-evaluation) and one
generation span for the final answer, with real inputs/outputs and
timing — so you can open Langfuse and see exactly how a given
architecture actually moved from query to answer.

Without tracing configured, every architecture behaves identically: this
module only ever observes a run, it never changes retrieval or
generation. Any tracing failure (bad keys, unreachable host) is swallowed
so a broken trace can never break the underlying RAG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TracingConfig:
    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"


class _NoopObservation:
    """Stands in for a real Langfuse span/generation when tracing is off
    or fails to initialize — every call is a safe no-op."""

    def update(self, **_kwargs: Any) -> "_NoopObservation":
        return self

    def end(self, **_kwargs: Any) -> "_NoopObservation":
        return self

    def start_observation(self, **_kwargs: Any) -> "_NoopObservation":
        return self


_NOOP = _NoopObservation()


class Tracer:
    """One Langfuse trace for a single architecture run.

    Usage:
        tracer = Tracer(tracing_config, name="Classic RAG run", query=query)
        span = tracer.step("vector_search", as_type="retriever", input=query)
        ...
        span.update(output=hits)
        span.end()
        trace_url = tracer.finish(output=answer)
    """

    def __init__(self, config: Optional[TracingConfig], name: str, query: str):
        self._client = None
        self._root = _NOOP
        if config is None:
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=config.public_key, secret_key=config.secret_key, host=config.host
            )
            self._root = self._client.start_observation(
                name=name, as_type="span", input={"query": query}
            )
        except Exception:  # noqa: BLE001 - tracing must never break a run
            self._client = None
            self._root = _NOOP

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def step(self, name: str, as_type: str = "span", input: Any = None, model: Optional[str] = None):
        """Start a child observation under this run's root span."""
        try:
            return self._root.start_observation(name=name, as_type=as_type, input=input, model=model)
        except Exception:  # noqa: BLE001
            return _NOOP

    def finish(self, output: Any = None) -> Optional[str]:
        """Close the root span and return a link to the trace, if available."""
        try:
            self._root.update(output=output)
            self._root.end()
        except Exception:  # noqa: BLE001
            pass

        trace_url = None
        if self._client is not None:
            try:
                trace_url = self._client.get_trace_url(trace_id=getattr(self._root, "trace_id", None))
            except Exception:  # noqa: BLE001
                trace_url = None
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                pass
        return trace_url
