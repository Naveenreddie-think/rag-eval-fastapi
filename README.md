# RAG System with Real Evaluation

Status: skeleton only -- build not yet started.

Corpus: FastAPI documentation (Tutorial + Advanced User Guide subset,
raw markdown, `tiangolo/fastapi` repo).

## Planned build stages
1. Corpus + basic ingestion
2. Embedding + vector store (Qdrant)
3. Hybrid retrieval (dense + BM25) + reranking
4. Hand-built evaluation set (75-100 QA pairs)
5. Evaluation methodology (retrieval precision/recall, faithfulness,
   ablation sweep across chunk size / embedding model / retrieval mode)
6. Security tie-in: poisoned-document test (reuses Project 1's attack
   taxonomy/methodology)
7. Serving (FastAPI + Streamlit) + deployment

See `FINDINGS.md` for design decisions and results as they're produced.

## Running locally (Windows)
Run modules with `python -m ...` from the project root, not
`python path/to/file.py`, to avoid `ModuleNotFoundError` from internal
`from app.x import y` imports:

```
python -m scripts.ingest
python -m scripts.run_eval
python -m app.api.main
```
