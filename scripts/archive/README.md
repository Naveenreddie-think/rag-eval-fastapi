# Archive

One-off historical investigation scripts, not part of the normal
ingestion/retrieval/eval workflow in `scripts/`. Each was written to
answer a specific question that came up during Steps 3-5, and each
finding it produced is already written up in `FINDINGS.md`.

Kept for provenance/transparency -- so the exact code behind a claim in
FINDINGS.md is still runnable and inspectable -- not because they're
expected to be run again or maintained going forward. None of these are
imported by app code, other scripts, or tests.

| Script | Finding it produced (see FINDINGS.md) |
|---|---|
| `diagnose_hybrid_regression.py` | Hybrid+rerank underperforming dense-only; cross-encoder demoting canonical chunks |
| `inspect_reranker_demotion.py` | Reranker header/vocabulary-clustering bias (sh_009, mh_012 cases) |
| `inspect_multihop_failures.py` | Multi-hop recall degradation analysis |
| `inspect_ragas_disagreement.py` | RAGAS vs. custom faithfulness metric biggest-disagreement cases |
| `check_mh015_retrieval.py` | mh_015 retrieval investigation (local model over-refusal) |
| `inspect_mh015_refusal.py` | mh_015/mh_025 over-refusal root cause |
| `inspect_local_model_regressions.py` | Qwen2.5-3B vs. Claude generation regressions |
| `rescore_with_fixed_refusal_check.py` | Refusal-detection fix re-scoring |
| `blind_recheck_sample.py` | Step 4 blind re-check of 10/80 QA pairs (seed=42) |
