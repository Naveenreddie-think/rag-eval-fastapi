# FastAPI Documentation RAG System with Real Evaluation

An empirically evaluated, structure-aware Retrieval-Augmented Generation (RAG) system built over the official FastAPI documentation (Tutorial and Advanced User Guide).

## Build Stages (All Complete ✅)
1. **Corpus + Ingestion** ✅: Sparse-checkout git loader with automatic `{ * docs_src/... * }` snippet resolution and fence-aware markdown header chunker.
2. **Embedding + Vector Store** ✅: `BAAI/bge-m3` dense embeddings (1024d) indexed in Qdrant Cloud.
3. **Hybrid Retrieval + Reranking** ✅: Rank-BM25 + Dense Qdrant fused via RRF ($k=60$) and reranked using `ms-marco-MiniLM-L-6-v2` with alpha score blending ($\alpha=0.7$).
4. **Hand-Built Evaluation Dataset** ✅: 80 curated QA pairs across `single_hop` (40), `multi_hop` (25), and `no_answer` (15) with exact source anchor mappings.
5. **Evaluation Methodology** ✅: Hit Rate@k, Recall@k, Precision@k, MRR, explainable atomic claim faithfulness metric, multi-dimensional ablation runner, and comparison against RAGAS framework.
6. **Security Evaluation** ✅: Poisoned-document indirect prompt injection and insecure guidance test suite.
7. **Serving & User Interface** ✅: Production FastAPI REST backend (`/api/query`, `/api/health`, `/api/metrics`) and interactive Streamlit UI with real-time latency diagnostics.
8. **Domain-Specific Fine-Tuning** ✅: Domain-adaptation module (`scripts/train_reranker.py`) mining hard negatives and fine-tuning cross-encoders on technical documentation.
9. **Docker & Cloud Deployment** ✅: Multi-service GPU-accelerated `docker-compose.yml` (`api` on 8000 + `ui` on 8501) and Streamlit Cloud configuration.

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
# Multi-dimensional ablation sweep (Chunk strategy x Embedding model x Retrieval mode)
python -m scripts.run_ablation_sweep

# Full 2-phase generation & faithfulness evaluation
python -m scripts.run_generation_eval

# Security poisoned-document attack evaluation
python -m scripts.run_security_eval

# Fine-tune domain-specific Cross-Encoder reranker
python -m scripts.train_reranker
```

### 4. Start Serving Backend & Interactive UI
```powershell
# Option A: Run directly in terminal
python -m app.api.main
python -m streamlit run streamlit_app.py

# Option B: Run via Docker Compose (FastAPI on 8000 + Streamlit on 8501)
docker compose up --build
```
