"""Data model shared by every RAG architecture module.

Each architecture (classic, hybrid, graph, agentic, corrective, multimodal)
defines exactly one ``ArchitectureSpec`` named ``SPEC`` in its own module.
Pages and the registry consume that spec — no architecture-specific layout
code lives outside the architecture's own module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# A single box in a pipeline diagram: (label, hex color). Labels may contain
# "<br>" for a line break inside the box.
Stage = Tuple[str, str]


@dataclass(frozen=True)
class PipelineSegment:
    """One visual row of a pipeline diagram.

    ``kind`` is either "row" (boxes connected left-to-right with arrows) or
    "parallel" (boxes rendered side by side in separate columns, used when
    a stage fans out into independent branches, e.g. Hybrid RAG's semantic
    + keyword search).
    """

    kind: str
    stages: Tuple[Stage, ...] | Tuple[Tuple[Stage, ...], ...]
    caption: Optional[str] = None


@dataclass(frozen=True)
class SliderControl:
    """An optional extra control shown in a page's "Try it" section."""

    key: str
    label: str
    min_value: int
    max_value: int
    default: int


@dataclass
class SimResult:
    """The real output of one run of an architecture's retrieval + answer pipeline."""

    steps: List[str]
    answer: str
    image_bytes: Optional[bytes] = None
    image_caption: Optional[str] = None
    dataframe: Optional[Any] = None
    graph_image_bytes: Optional[bytes] = None
    trace_url: Optional[str] = None


@dataclass(frozen=True)
class ArchitectureSpec:
    key: str
    name: str
    tagline: str
    icon: str
    description: str
    pipeline: List[PipelineSegment]
    example_quote: str
    good_points: List[str]
    bad_points: List[str]
    default_query: str
    simulate: Callable[..., SimResult]
    compare_row: Dict[str, str]
    slider: Optional[SliderControl] = None
