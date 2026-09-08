"""Corrective RAG — Check.

Retrieves normally, then actually grades each retrieved chunk's relevance
by cosine similarity to the query, drops anything below threshold, and
supplements with a real web search if too little survives — retrieval
returning results is only the beginning; those results still need to
support the question.
"""

from __future__ import annotations

from typing import Optional

from rag_compare.architectures.base import ArchitectureSpec, PipelineSegment, SimResult
from rag_compare.corpus import DOCUMENTS
from rag_compare.llm import LlmConfig, generate_answer, model_label
from rag_compare.retrieval.vector_store import TfidfVectorStore
from rag_compare.retrieval.web_search import duckduckgo_search
from rag_compare.tracing import Tracer, TracingConfig

_store = TfidfVectorStore(DOCUMENTS)

RELEVANCE_THRESHOLD = 0.12
MIN_SURVIVING_CHUNKS = 2


def simulate(
    query: str,
    llm_config: Optional[LlmConfig] = None,
    tracing_config: Optional[TracingConfig] = None,
) -> SimResult:
    tracer = Tracer(tracing_config, name="Corrective RAG run", query=query)
    steps = [f"1. Query: \"{query}\"", "2. Retrieve top-k chunks from the vector DB:"]

    search_span = tracer.step("vector_search", as_type="retriever", input=query)
    hits = _store.search(query, k=3)
    search_span.update(output=[{"title": h["doc"]["title"], "score": h["score"]} for h in hits])
    search_span.end()
    for h in hits:
        steps.append(f"    • score={h['score']:.3f} — {h['doc']['title']}")

    grade_span = tracer.step(
        "relevance_grading", as_type="evaluator", input={"threshold": RELEVANCE_THRESHOLD}
    )
    steps.append(f"3. Evaluate relevance of each chunk (cosine score >= {RELEVANCE_THRESHOLD} to keep):")
    kept = []
    grading_output = []
    for h in hits:
        verdict = "KEEP" if h["score"] >= RELEVANCE_THRESHOLD else "DROP"
        steps.append(f"    • {verdict} (score={h['score']:.3f}) — {h['doc']['title']}")
        grading_output.append({"title": h["doc"]["title"], "score": h["score"], "verdict": verdict})
        if verdict == "KEEP":
            kept.append(h)
    grade_span.update(output=grading_output)
    grade_span.end()

    if len(kept) < MIN_SURVIVING_CHUNKS:
        steps.append(
            f"4. Only {len(kept)} chunk(s) survived grading (< {MIN_SURVIVING_CHUNKS}) — "
            "supplementing with a real web search."
        )
        supplement_span = tracer.step("web_search_supplement", as_type="tool", input=query)
        result = duckduckgo_search(query)
        supplement_span.update(output=result)
        supplement_span.end()
        if result["available"]:
            steps.append(f"    • [web search] {result['snippet'][:200]}")
            kept.append({"doc": {"id": "web", "title": f"Web result ({result['source']})", "text": result["snippet"]}, "score": 1.0})
        else:
            steps.append(f"    • [web search] {result['snippet']}")
    else:
        steps.append(f"4. {len(kept)} chunk(s) passed grading — no supplement needed.")

    context = "\n\n".join(f"{h['doc']['title']}: {h['doc']['text']}" for h in kept)
    gen_span = tracer.step(
        "generate_answer", as_type="generation", input=context, model=model_label(llm_config)
    )
    answer, used_llm = generate_answer(query, context, llm_config)
    gen_span.update(output=answer)
    gen_span.end()

    generator = "LLM generated" if used_llm else "Extractive fallback synthesized"
    steps.append(f"5. {generator} the answer only from chunks that passed the relevance check.")

    trace_url = tracer.finish(output=answer)
    return SimResult(steps=steps, answer=answer, trace_url=trace_url)


SPEC = ArchitectureSpec(
    key="corrective",
    name="Corrective RAG",
    tagline="Check",
    icon="🛠️",
    description=(
        "When retrieved evidence needs checking. Evaluates retrieved material, then "
        "refines or supplements it before generation — retrieval returning results is "
        "only the beginning. Those results still need to support the question."
    ),
    pipeline=[
        PipelineSegment(
            "row",
            (
                ("Query", "#2563eb"),
                ("Retrieve", "#7c3aed"),
                ("Evaluate<br>Relevance", "#b45309"),
                ("Refine /<br>Supplement", "#b91c1c"),
                ("LLM", "#0f766e"),
                ("Answer", "#059669"),
            ),
            caption=(
                "If evaluation finds the retrieved chunks weak or off-target, the loop "
                "back to \"Refine / Supplement\" re-queries, falls back to web search, "
                "or strips out irrelevant chunks before generation."
            ),
        )
    ],
    example_quote=(
        "Those results still need to support the question — a relevance grader sits "
        "between retrieval and generation instead of trusting Top-K blindly."
    ),
    good_points=[
        "Retrieval quality is inconsistent or the corpus is noisy",
        "Wrong or irrelevant context is costly (compliance, medical, legal, financial answers)",
        "You need a safety net that catches bad retrieval before it reaches the LLM",
    ],
    bad_points=[
        "Extra evaluation step adds latency and an additional scoring/grading pass",
        "Grader quality becomes a new failure point — a bad threshold can discard good context",
        "Unnecessary overhead when the corpus is small, clean, and retrieval is already reliable",
    ],
    default_query="What are the side effects of Drug X at a 20mg dose?",
    simulate=simulate,
    compare_row={
        "Core idea": "Check",
        "Retrieval hops": "Single-hop + evaluation loop",
        "Data source(s)": "Vector DB (+ web fallback)",
        "Adapts mid-query": "Partially (on evaluation failure)",
        "Self-correcting": "Yes",
        "Latency / cost": "Medium",
        "Best for": "Cases where wrong context is costly",
        "Main weakness": "Grader adds latency and is a new failure point",
    },
)
