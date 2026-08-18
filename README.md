# FastAPI Documentation RAG System with Real Evaluation

An empirically evaluated, structure-aware Retrieval-Augmented Generation (RAG) system built over the official FastAPI documentation (Tutorial and Advanced User Guide).

## Build Stages (All Complete ✅)
1. **Corpus + Ingestion** ✅: Sparse-checkout git loader with automatic `{ * docs_src/... * }` snippet resolution and fence-aware markdown header chunker.
2. **Embedding + Vector Store** ✅: `BAAI/bge-m3` dense embeddings (1024d) indexed in Qdrant Cloud.
3. **Hybrid Retrieval + Reranking** ✅: Rank-BM25 + Dense Qdrant fused via RRF ($k=60$) and reranked using `ms-marco-MiniLM-L-6-v2` with alpha score blending ($\alpha=0.7$).
4. **Hand-Built Evaluation Dataset** ✅: 80 curated QA pairs across `single_hop` (50), `multi_hop` (15), and `no_answer` (15) with exact source anchor mappings.
5. **Evaluation Methodology** ✅: Hit Rate@k, Recall@k, Precision@k, MRR, explainable atomic claim faithfulness metric, and comparison against RAGAS framework.
6. **Security Evaluation** ✅: Poisoned-document indirect prompt injection and insecure guidance test suite.
7. **Serving & User Interface** ✅: Production FastAPI REST backend (`/api/query`, `/api/health`, `/api/metrics`) and interactive Streamlit UI with real-time latency diagnostics.

See `FINDINGS.md` for measured benchmarks, failure analyses, and architectural decisions.

---

## Running Locally

Always run modules with `python -m ...` from the project root:

### 1. Ingestion & Vector Indexing
```powershell
# Chunk markdown documents to data/processed/chunks.json
python -m scripts.ingest

# (Optional) Ingest and sync to Qdrant Cloud collection
python -m scripts.ingest --sync-qdrant
```

### 2. Run Automated Test Suite
```powershell
python -m pytest tests/ -v
```

### 3. Run Evaluations & Benchmarks
```powershell
# Unified retrieval ablation sweep (Dense vs BM25 vs Hybrid variants)
python -m scripts.run_eval

# Full 2-phase generation & faithfulness evaluation
python -m scripts.run_generation_eval

# Security poisoned-document attack evaluation
python -m scripts.run_security_eval
```

### 4. Start Serving Backend & Interactive UI
```powershell
# Start FastAPI REST API (http://localhost:8000/docs)
python -m app.api.main

# Launch Streamlit Web UI (in a new terminal)
python -m streamlit run streamlit_app.py
```
