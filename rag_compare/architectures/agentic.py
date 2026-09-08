"""Agentic RAG — Reason.

A real (if small) agent loop: each round it picks a tool (vector DB, then
web search, then knowledge graph), retrieves, appends to its running
context, and computes a keyword-coverage self-evaluation score against the
query. It stops as soon as coverage clears a confidence threshold or the
round budget runs out — the loop actually adapts to what it has already
found instead of running a fixed number of times.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import (
    ArchitectureSpec,
    PipelineSegment,
    SimResult,
    SliderControl,
)
from rag_compare.corpus import DOCUMENTS, GRAPH_TRIPLES
from rag_compare.llm import LlmConfig, generate_answer
from rag_compare.retrieval.graph_store import KnowledgeGraph
from rag_compare.retrieval.vector_store import TfidfVectorStore
from rag_compare.retrieval.web_search import duckduckgo_search
from rag_compare.scoring import keyword_coverage

_store = TfidfVectorStore(DOCUMENTS)
_graph = KnowledgeGraph(GRAPH_TRIPLES)

CONFIDENCE_THRESHOLD = 0.65
TOOL_ORDER = ["vector_db", "web_search", "knowledge_graph"]


def simulate(query: str, llm_config: Optional[LlmConfig] = None, loops: int = 2) -> SimResult:
    steps = [f"1. Query: \"{query}\"", "2. Agent plans a tool budget of up to " f"{loops} round(s)."]

    context_parts: list[str] = []
    coverage = 0.0

    for round_num in range(1, loops + 1):
        tool = TOOL_ORDER[(round_num - 1) % len(TOOL_ORDER)]
        steps.append(f"**Round {round_num} — agent selects tool: `{tool}`**")

        if tool == "vector_db":
            hits = _store.search(query, k=2)
            for h in hits:
                steps.append(f"    • [vector DB] score={h['score']:.3f} — {h['doc']['title']}")
                context_parts.append(f"{h['doc']['title']}: {h['doc']['text']}")

        elif tool == "web_search":
            result = duckduckgo_search(query)
            if result["available"]:
                steps.append(f"    • [web search] {result['snippet'][:200]}")
                context_parts.append(result["snippet"])
            else:
                steps.append(f"    • [web search] {result['snippet']} Falling back to cached internal benchmark doc.")
                cached = next((d for d in DOCUMENTS if d["id"] == "finance-benchmark"), None)
                if cached:
                    context_parts.append(f"{cached['title']}: {cached['text']}")

        elif tool == "knowledge_graph":
            entities = _graph.extract_entities(query)
            triples = _graph.connected_context(entities, hops=2) if entities else []
            if triples:
                for s, rel, o in triples:
                    steps.append(f"    • [graph] ({s}) —[{rel}]→ ({o})")
                    context_parts.append(f"{s} {rel} {o}.")
            else:
                steps.append("    • [graph] no matching entities in this query")

        context_so_far = "\n".join(context_parts)
        coverage = keyword_coverage(query, context_so_far)
        steps.append(f"    Self-evaluation: keyword coverage = {coverage:.0%} (need >= {CONFIDENCE_THRESHOLD:.0%})")

        if coverage >= CONFIDENCE_THRESHOLD:
            steps.append(f"    Confidence threshold cleared — stopping after round {round_num}/{loops}.")
            break
    else:
        steps.append(f"    Round budget of {loops} exhausted before reaching the confidence threshold.")

    context = "\n\n".join(context_parts)
    answer, used_llm = generate_answer(query, context, llm_config)
    generator = "LLM generated" if used_llm else "Extractive fallback synthesized"
    steps.append(f"3. {generator} the final answer from everything gathered across all rounds.")

    return SimResult(steps=steps, answer=answer)


SPEC = ArchitectureSpec(
    key="agentic",
    name="Agentic RAG",
    tagline="Reason",
    icon="🤖",
    description=(
        "Adaptive, iterative, self-correcting. The agent itself decides what to "
        "retrieve, where to look, and whether to loop back for another pass."
    ),
    pipeline=[
        PipelineSegment("row", (("Query", "#2563eb"), ("Reasoning<br>Agent", "#7c3aed"))),
        PipelineSegment(
            "row",
            (
                ("Vector DB", "#0f766e"),
                ("Knowledge<br>Graph", "#b91c1c"),
                ("Web Search", "#b45309"),
                ("Tools (e.g. SQL)", "#334155"),
            ),
            caption="↑ the agent chooses one, several, or repeated rounds of the above ↑",
        ),
        PipelineSegment("row", (("Self-Evaluation", "#7c3aed"), ("Final Answer", "#059669"))),
    ],
    example_quote=(
        "One question might require vector search, SQL, <em>and</em> a web lookup. "
        "Give that loop clear stopping conditions and a budget — otherwise it will "
        "keep looping."
    ),
    good_points=[
        "The query is complex enough to need multiple retrieval rounds",
        "Source-switching is required (vector DB isn't enough — needs SQL, web, or other tools too)",
        "A quality check before the answer ships matters (high-stakes or ambiguous asks)",
    ],
    bad_points=[
        "Higher latency and cost — multiple LLM calls and tool round-trips per query",
        "Harder to debug/predict — the agent's path isn't fixed ahead of time",
        "Without clear stopping conditions and a budget, the loop can run away",
    ],
    default_query="Compare our Q3 revenue trend to the latest industry benchmark",
    simulate=simulate,
    compare_row={
        "Core idea": "Reason",
        "Retrieval hops": "Iterative, agent-decided",
        "Data source(s)": "Vector DB + Knowledge Graph + Web + Tools",
        "Adapts mid-query": "Yes",
        "Self-correcting": "Yes",
        "Latency / cost": "High",
        "Best for": "Complex, multi-step, high-stakes queries",
        "Main weakness": "Slowest, most expensive; needs stopping conditions & budget",
    },
    slider=SliderControl(key="loops", label="Max reasoning rounds", min_value=1, max_value=3, default=2),
)
