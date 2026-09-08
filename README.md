# modern-rag-architectures

An interactive Streamlit app comparing six modern Retrieval-Augmented Generation
(RAG) architectures, each in its own section:

- **Classic RAG** — Retrieve: single-hop semantic search over a vector DB
- **Hybrid RAG** — Combine: semantic search + keyword search, merged and re-ranked
- **Graph RAG** — Connect: entity extraction + knowledge graph traversal
- **Agentic RAG** — Reason: an agent iterates across vector DB, graph, web, and tools
- **Corrective RAG** — Check: evaluates and refines retrieved evidence before generation
- **Multimodal RAG** — Expand: retrieves across text, images, charts, and tables

Each section shows the pipeline as a flow diagram, when it works well vs. where it
breaks, and a simulated step-by-step walkthrough you can run with your own query.
A "Compare All" tab lays every architecture side by side in one table.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (defaults to `http://localhost:8501`).
