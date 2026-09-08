# modern-rag-architectures

Six **working** Retrieval-Augmented Generation apps — Classic, Hybrid, Graph,
Agentic, Corrective, and Multimodal RAG — each its own Streamlit page, all
sharing one real retrieval stack. There is no scripted demo text: every page
runs actual retrieval against a real corpus, and you can type your own
question.

## What's real here

| Piece | Implementation |
|---|---|
| Vector database | **FAISS** (`IndexFlatIP`) over TF-IDF embeddings computed locally with scikit-learn — no model download, fully offline |
| Keyword search | **BM25** (`rank_bm25`) for exact-match terms like error codes |
| Hybrid re-ranking | **Reciprocal Rank Fusion** merging the semantic and keyword rankings |
| Knowledge graph | **networkx** `MultiDiGraph`, with real entity extraction and multi-hop traversal, rendered to an actual graph image |
| Web search | Live call to the DuckDuckGo Instant Answer API (no key required), with a graceful offline fallback |
| Relevance grading | Real cosine-similarity thresholding (Corrective RAG) and keyword-coverage self-evaluation (Agentic RAG) |
| Generation | **Claude** (`claude-sonnet-5`) via the Anthropic SDK if you supply an API key in the sidebar; otherwise a real extractive summarizer (term-overlap sentence scoring) — retrieval is real either way |
| Multimodal | A chart and table rendered from real numbers with matplotlib/pandas; Claude's vision API reads the chart image directly when a key is supplied |

Bring your own Anthropic API key (optional, entered in the sidebar, never
stored) to get real Claude-generated answers and real chart reading. Without
one, every page still runs end-to-end on the extractive fallback.

## Project structure

```
modern-rag-architectures/
├── app.py                        # Landing page: overview + links to each app
├── pages/                        # One Streamlit page per architecture
│   ├── 1_Classic_RAG.py
│   ├── 2_Hybrid_RAG.py
│   ├── 3_Graph_RAG.py
│   ├── 4_Agentic_RAG.py
│   ├── 5_Corrective_RAG.py
│   ├── 6_Multimodal_RAG.py
│   └── 7_Compare_All.py          # Side-by-side comparison table
├── rag_compare/                  # Shared package — all real logic lives here
│   ├── corpus.py                 # The demo knowledge base (docs, graph triples, revenue data)
│   ├── multimodal_data.py        # Real chart/table generation from the revenue data
│   ├── scoring.py                # Keyword-coverage / relevance heuristics
│   ├── llm.py                    # Optional Claude generation + extractive fallback
│   ├── ui.py                     # Shared page layout: pipeline diagrams, sidebar, results
│   ├── registry.py               # Single source of truth: all architecture specs, in order
│   ├── retrieval/                # The real retrieval backends
│   │   ├── vector_store.py       # FAISS + TF-IDF vector database
│   │   ├── keyword_store.py      # BM25 keyword search
│   │   ├── fusion.py             # Reciprocal rank fusion
│   │   ├── graph_store.py        # networkx knowledge graph
│   │   └── web_search.py         # DuckDuckGo web search
│   └── architectures/            # One module per architecture: pipeline + simulate()
│       ├── base.py               # ArchitectureSpec / SimResult data model
│       ├── classic.py
│       ├── hybrid.py
│       ├── graph.py
│       ├── agentic.py
│       ├── corrective.py
│       └── multimodal.py
├── tests/                        # pytest suite covering retrieval + registry
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml
```

Each architecture is a self-contained module: its pipeline diagram, when it
works well vs. where it breaks, and a `simulate(query, ...)` function that
runs the real retrieval + answer pipeline. `rag_compare/ui.py` renders any
of them the same way, so a page file is just three lines wiring a spec into
the shared layout.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (defaults to `http://localhost:8501`) and
use the sidebar to navigate between architectures, or click through from the
landing page.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```
