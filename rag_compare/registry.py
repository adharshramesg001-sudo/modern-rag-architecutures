"""Single source of truth for which architectures exist and their order."""

from __future__ import annotations

from rag_compare.architectures import agentic, classic, corrective, graph, hybrid, multimodal
from rag_compare.architectures.base import ArchitectureSpec

ALL_SPECS: list[ArchitectureSpec] = [
    classic.SPEC,
    hybrid.SPEC,
    graph.SPEC,
    agentic.SPEC,
    corrective.SPEC,
    multimodal.SPEC,
]

BY_KEY: dict[str, ArchitectureSpec] = {spec.key: spec for spec in ALL_SPECS}
