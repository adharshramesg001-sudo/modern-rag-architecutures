"""Graph RAG — Connect.

Extracts entities mentioned in the query, locates them as real nodes in a
networkx knowledge graph, walks the edges connecting them, and builds
context from the traversed subgraph — the evidence a pure vector search
over isolated chunks would never assemble.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.corpus import GRAPH_TRIPLES
from rag_compare.llm import LlmConfig, generate_answer, model_label
from rag_compare.retrieval.graph_store import KnowledgeGraph
from rag_compare.tracing import Tracer, TracingConfig

_graph = KnowledgeGraph(GRAPH_TRIPLES)


def simulate(
    query: str,
    llm_config: Optional[LlmConfig] = None,
    tracing_config: Optional[TracingConfig] = None,
) -> SimResult:
    tracer = Tracer(tracing_config, name="Graph RAG run", query=query)
    steps = [f"1. Query: \"{query}\""]

    entity_span = tracer.step("entity_extraction", input=query)
    entities = _graph.extract_entities(query)
    entity_span.update(output=entities)
    entity_span.end()
    steps.append(f"2. Entity extraction against the graph's {len(_graph.entities)} known nodes: " + (
        ", ".join(f"`{e}`" for e in entities) if entities else "none matched"
    ))

    traverse_span = tracer.step("graph_traversal", as_type="retriever", input=entities)
    triples = _graph.connected_context(entities, hops=2) if entities else []
    traverse_span.update(output=[f"({s}) -[{rel}]-> ({o})" for s, rel, o in triples])
    traverse_span.end()
    steps.append(f"3. Walk the graph 2 hops out from each matched entity — {len(triples)} connected edge(s) found:")
    for s, rel, o in triples:
        steps.append(f"    • ({s}) —[{rel}]→ ({o})")

    context = "\n".join(f"{s} {rel} {o}." for s, rel, o in triples)
    gen_span = tracer.step(
        "generate_answer", as_type="generation", input=context, model=model_label(llm_config)
    )
    answer, used_llm = generate_answer(query, context, llm_config)
    gen_span.update(output=answer)
    gen_span.end()

    generator = "LLM generated" if used_llm else "Extractive fallback synthesized"
    steps.append(f"4. {generator} the answer from the connected subgraph (not a single chunk).")

    highlight = set(entities)
    for s, _, o in triples:
        highlight.add(s)
        highlight.add(o)
    graph_png = _graph.render_png(highlight_nodes=list(highlight)) if triples else _graph.render_png()

    trace_url = tracer.finish(output=answer)
    return SimResult(steps=steps, answer=answer, graph_image_bytes=graph_png, trace_url=trace_url)


SPEC = ArchitectureSpec(
    key="graph",
    name="Graph RAG",
    tagline="Connect",
    icon="🕸️",
    description=(
        "Relationship-driven, entity-heavy, pulls from multiple sources. Rather than "
        "pulling similar-looking chunks, it identifies entities in the query, walks "
        "the graph connecting them, and builds context around those relationships — "
        "something plain vector search can't do."
    ),
    pipeline=[
        PipelineSegment(
            "row",
            (
                ("Query", "#2563eb"),
                ("Entity<br>Extraction", "#b45309"),
                ("Knowledge<br>Graph", "#b91c1c"),
                ("Connected<br>Context", "#b91c1c"),
                ("LLM", "#0f766e"),
                ("Answer", "#059669"),
            ),
        )
    ],
    example_quote=(
        "&#8220;Which projects depend on suppliers affected by this incident?&#8221; — "
        "the evidence lives across connected facts, not inside any single chunk."
    ),
    good_points=[
        "The answer isn't in one document — it's in how entities link to each other across your data",
        "Multi-hop questions (\"how is X related to Y through Z?\")",
        "Domains with rich, explicit relationships: orgs, people, products, supply chains",
    ],
    bad_points=[
        "Requires building and maintaining a knowledge graph up front — real cost and effort",
        "Entity extraction quality caps answer quality (missed/mislabeled entities = missed links)",
        "Overkill for simple, single-document lookups that plain vector search already solves",
    ],
    default_query="How is Acme Corp connected to the FDA recall, and which projects are exposed?",
    simulate=simulate,
    compare_row={
        "Core idea": "Connect",
        "Retrieval hops": "Multi-hop via graph edges",
        "Data source(s)": "Knowledge Graph (+ sources it indexes)",
        "Adapts mid-query": "No",
        "Self-correcting": "No",
        "Latency / cost": "Medium",
        "Best for": "Answers that depend on how entities relate",
        "Main weakness": "Needs a maintained knowledge graph",
    },
)
