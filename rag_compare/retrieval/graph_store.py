"""A real knowledge graph (networkx) with entity extraction and relationship
traversal — the mechanism behind Graph RAG and part of Agentic RAG's toolset."""

from __future__ import annotations

import io
from typing import List, Set, Tuple

import networkx as nx


class KnowledgeGraph:
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = triples
        self.graph = nx.MultiDiGraph()
        for subject, relation, obj in triples:
            self.graph.add_edge(subject, obj, relation=relation)
        self.entities: List[str] = sorted(self.graph.nodes())

    def extract_entities(self, query: str) -> List[str]:
        """Real (if simple) named-entity matching: match known graph node
        names against the query text, longest names first so e.g. "Project
        Falcon" wins over a looser partial match."""
        q = query.lower()
        found = [e for e in sorted(self.entities, key=len, reverse=True) if e.lower() in q]
        return found

    def connected_context(self, entities: List[str], hops: int = 2) -> List[Tuple[str, str, str]]:
        """Walk the graph outward from each matched entity and return every
        edge inside that neighborhood — the "connected context" a query
        needs when the answer spans more than one document."""
        nodes: Set[str] = set()
        for entity in entities:
            if entity in self.graph:
                nodes |= set(nx.ego_graph(self.graph.to_undirected(), entity, radius=hops).nodes())
        if not nodes:
            return []
        subgraph = self.graph.subgraph(nodes)
        return [(u, data["relation"], v) for u, v, data in subgraph.edges(data=True)]

    def render_png(self, highlight_nodes: List[str] | None = None) -> bytes:
        """Render the full graph (or just the traversed neighborhood) to PNG bytes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        highlight = set(highlight_nodes or [])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pos = nx.spring_layout(self.graph, seed=7)
        node_colors = ["#f59e0b" if n in highlight else "#93c5fd" for n in self.graph.nodes()]
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=1200, ax=ax)
        nx.draw_networkx_labels(self.graph, pos, font_size=7, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, ax=ax, alpha=0.6, arrowsize=12)
        edge_labels = {(u, v): d["relation"] for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)
        ax.axis("off")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return buf.getvalue()
