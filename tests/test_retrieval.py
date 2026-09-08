from rag_compare.corpus import DOCUMENTS, GRAPH_TRIPLES
from rag_compare.retrieval.fusion import reciprocal_rank_fusion
from rag_compare.retrieval.graph_store import KnowledgeGraph
from rag_compare.retrieval.keyword_store import Bm25Store
from rag_compare.retrieval.vector_store import TfidfVectorStore
from rag_compare.scoring import keyword_coverage


def test_vector_store_finds_the_relevant_refund_doc():
    store = TfidfVectorStore(DOCUMENTS)
    hits = store.search("What is the refund policy?", k=3)
    assert hits
    assert hits[0]["doc"]["id"] == "policy-refund"
    assert hits[0]["score"] > 0


def test_bm25_finds_exact_error_code():
    store = Bm25Store(DOCUMENTS)
    hits = store.search("error E104", k=3)
    assert hits
    assert hits[0]["doc"]["id"] == "troubleshoot-e104"


def test_reciprocal_rank_fusion_merges_two_rankings():
    docs_a = [{"doc": DOCUMENTS[0], "score": 0.9}, {"doc": DOCUMENTS[1], "score": 0.5}]
    docs_b = [{"doc": DOCUMENTS[1], "score": 5.0}, {"doc": DOCUMENTS[0], "score": 1.0}]
    fused = reciprocal_rank_fusion([docs_a, docs_b], top_n=2)
    assert {h["doc"]["id"] for h in fused} == {DOCUMENTS[0]["id"], DOCUMENTS[1]["id"]}


def test_knowledge_graph_traversal_connects_acme_to_projects():
    graph = KnowledgeGraph(GRAPH_TRIPLES)
    entities = graph.extract_entities("How is Acme Corp connected to the FDA recall?")
    assert "Acme Corp" in entities
    assert "FDA" in entities

    triples = graph.connected_context(entities, hops=3)
    touched_nodes = {s for s, _, _ in triples} | {o for _, _, o in triples}
    assert "Project Falcon" in touched_nodes or "Project Nova" in touched_nodes


def test_keyword_coverage_is_higher_with_more_matching_context():
    query = "Q3 revenue benchmark"
    low = keyword_coverage(query, "The cafeteria added a salad bar.")
    high = keyword_coverage(query, "Q3 revenue reached new highs versus the industry benchmark.")
    assert high > low
