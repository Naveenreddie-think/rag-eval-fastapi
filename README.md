---
title: FastAPI Docs RAG Evaluator


sdk: docker
app_port: 8501
pinned: false
---

# FastAPI Docs RAG Evaluation Engine

A retrieval-augmented generation system built over FastAPI's real documentation, with retrieval quality, answer faithfulness, and refusal behavior all measured directly rather than assumed.

## The problem

Developers reach for an AI assistant over documentation because reading through pages of docs to find one answer is slow. But a RAG system that just retrieves *something* and generates a plausible-sounding answer is actively dangerous in a technical setting: a hallucinated code example, a wrong parameter name, or a confidently wrong answer to something the docs don't actually cover can cost more time than it saves, and erodes trust the moment it's caught.

For a documentation assistant to be genuinely usable, three things have to be true, and provably so: it has to retrieve the right source material, the generated answer has to actually be grounded in what was retrieved (not just plausible-sounding), and it has to recognize when a question falls outside what the documentation covers, rather than guessing.

This project builds a RAG system over FastAPI's real documentation and measures all three of those properties directly, rather than assuming they hold. Every retrieval configuration, every reranking decision, every generation model swap is backed by a measured result against a hand-verified 80-pair evaluation set.

## What this does

A RAG system over FastAPI's real documentation (the Tutorial and Advanced User Guide): hybrid retrieval with cross-encoder reranking, grounded generation with an explainable faithfulness metric, a reranker fine-tuned on this project's own corpus, a security check against poisoned documents, and a FastAPI + Streamlit serving layer -- all measured, with failure modes documented as honestly as the successes. `FINDINGS.md` has the full account, including several things that didn't work as expected and had to be root-caused rather than assumed away.

## Architecture

```
FastAPI docs (git sparse-checkout)
        |
        v
  Ingestion & Chunking        -- resolve snippet-include markers, split on
                                  markdown headers, fence-aware (no code
                                  block is ever split mid-fence)
        |
        v
  Embedding (BGE-M3, GPU)      -- dense vectors indexed in Qdrant Cloud
        |
        v
  Hybrid Retrieval             -- BM25 (sparse) + dense search, fused via
                                  Reciprocal Rank Fusion
        |
        v
  Cross-Encoder Reranking      -- blended with the RRF score (alpha=0.7);
                                  fine-tuned on this corpus's own hard
                                  negatives
        |
        v
  Grounded Generation           -- Qwen2.5-3B-Instruct (local), constrained
                                  to answer only from retrieved context
        |
        v
  Evaluation                    -- retrieval precision/recall/MRR, claim-
                                  level faithfulness, refusal accuracy,
                                  security (poisoned-document) testing
```

Each stage is a real, independently swappable module (`app/ingestion/`, `app/embeddings/`, `app/retrieval/`, `app/generation/`, `app/eval/`) -- retrieval mode, blend weight, and generation model are all runtime-configurable, not hardcoded to one path.

## Key findings

The full account, with root-cause diagnoses and honest limitations, is in `FINDINGS.md`. The headline numbers:

- Dense-only retrieval (BGE-M3) hits 0.831 Hit Rate@5 / 0.692 Recall@5 across 65 ground-truth queries -- and outperforms naive hybrid retrieval on both, because the off-the-shelf reranker turned out to be biased toward header-vocabulary overlap rather than actual relevance (diagnosed directly, not assumed -- see `FINDINGS.md` for the specific cases).
- Fine-tuning that same reranker on this corpus's own hard negatives fixed it: MRR moved from 0.624 to 0.739 (+18.4%) and Hit Rate@5 from 0.769 to 0.877 (+14.0%), measured against the live end-to-end pipeline, not just the reranker in isolation.
- The custom claim-decomposition faithfulness metric scored 0.971 (single-hop) / 0.914 (multi-hop) against Claude Sonnet 4.5, with 100% parsing success across all 65 pairs -- RAGAS's own equivalent metric failed to parse its own LLM output on 4 of those 65.
- Anthropic API credits ran out mid-evaluation. Rather than stop, generation moved to a local Qwen2.5-3B-Instruct model, which needed real, different reliability engineering that a hosted frontier model never did -- documented in full in `FINDINGS.md` -- and produced a genuine hosted-vs-local comparison instead of just working around the outage.

## Tech stack

**Retrieval**
- `BAAI/bge-m3` dense embeddings (1024d), GPU-accelerated
- Qdrant Cloud for vector storage
- Rank-BM25 for sparse/keyword retrieval
- `cross-encoder/ms-marco-MiniLM-L-6-v2`, both off-the-shelf and fine-tuned on this project's own corpus

**Generation**
- Qwen2.5-3B-Instruct (local, bf16, via `transformers`) -- previously Claude Sonnet 4.5

**Serving**
- FastAPI (`/api/query`, `/api/health`, `/api/metrics`)
- Streamlit (interactive UI)
- Docker / Docker Compose (multi-service, GPU passthrough)

**Evaluation**
- Custom retrieval metrics (Hit Rate@k, Recall@k, Precision@k, MRR)
- Custom claim-decomposition faithfulness metric
- RAGAS (comparison baseline)
- Hand-built 80-pair QA ground-truth set (single-hop, multi-hop, no-answer)

## Repo structure

- `app/` -- the actual pipeline: `ingestion/` (corpus loading + chunking), `embeddings/` (BGE-M3 wrapper), `retrieval/` (BM25, dense, hybrid), `generation/` (local LLM), `eval/` (metrics, faithfulness, QA dataset loading), `api/` (the FastAPI app).
- `scripts/` -- runnable entry points: ingestion, the eval suites (`run_eval`, `run_retrieval_eval`, `run_generation_eval`, `run_ablation_sweep`, `run_security_eval`, `run_ragas_comparison`), reranker fine-tuning (`train_reranker`), and one-off verification scripts. `scripts/archive/` holds one-off historical investigation scripts tied to specific findings already written up in `FINDINGS.md` -- kept for provenance, not part of the normal workflow.
- `eval/` -- the hand-built 80-pair QA ground-truth set (`qa_pairs.json`).
- `data/` -- `raw/` (the sparse-cloned FastAPI docs, gitignored, regenerated by ingestion) and `processed/` (the chunked corpus and cached eval results).
- `tests/` -- the automated test suite (ingestion, retrieval, eval metrics, API).

## Running it locally

Always run modules with `python -m ...` from the project root.

### 1. Ingestion & vector indexing
```powershell
# Chunk markdown documents to data/processed/chunks.json
python -m scripts.ingest

# (Optional) Ingest and sync to Qdrant Cloud collection
python -m scripts.ingest --sync-qdrant
```

### 2. Run the automated test suite
```powershell
python -m pytest tests/ -v
```

### 3. Run evaluations & benchmarks
```powershell
# Multi-dimensional ablation sweep (chunk strategy x embedding model x retrieval mode)
python -m scripts.run_ablation_sweep

# Full 2-phase generation & faithfulness evaluation
python -m scripts.run_generation_eval

# Security poisoned-document attack evaluation
python -m scripts.run_security_eval

# Fine-tune the domain-specific cross-encoder reranker
python -m scripts.train_reranker
```

### 4. Start the serving backend & interactive UI
```powershell
# Option A: run directly in terminal
python -m app.api.main
python -m streamlit run streamlit_app.py

# Option B: run via Docker Compose (FastAPI on 8000 + Streamlit on 8501)
docker compose up --build
```

---

The full technical writeup -- every design decision, every bug found and fixed, every measured result -- is in [`FINDINGS.md`](FINDINGS.md).
