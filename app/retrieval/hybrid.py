"""
Step 3: Hybrid retrieval + reranking.

Fuses dense + BM25 results via Reciprocal Rank Fusion (RRF), then
reranks the fused candidates with a cross-encoder for the final top-k.

RRF chosen over a weighted score combination because dense (cosine
similarity, ~0-1) and BM25 (unbounded, corpus-dependent) scores aren't
on comparable scales -- RRF sidesteps that by using rank position only,
not raw scores.

Cross-encoder reranking is a separate, more expensive pass: it sees the
actual (query, chunk_text) pair jointly, rather than comparing
precomputed independent embeddings, so it's more accurate but too slow
to run over the whole corpus -- only run over the fused candidate set.
"""

import time

from sentence_transformers import CrossEncoder

from app.retrieval.bm25 import bm25_search
from app.retrieval.dense import dense_search

_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_NAME, device="cuda")
    return _cross_encoder


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Standard RRF: score = sum(1 / (k + rank)) across all lists a chunk
    appears in. k=60 is the commonly used default constant from the
    original RRF paper -- dampens the impact of rank 1 vs rank 2 while
    still rewarding chunks that rank highly in either/both lists."""
    scores: dict[int, float] = {}
    item_lookup: dict[int, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            item_lookup[cid] = item

    fused_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [{**item_lookup[cid], "rrf_score": scores[cid]} for cid in fused_ids]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Cross-encoder rerank over a candidate set (NOT the whole corpus --
    too slow). Returns top_k candidates sorted by cross-encoder score."""
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = _get_cross_encoder().predict(pairs)
    scored = list(zip(candidates, ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in scored[:top_k]]


def hybrid_search(query: str, top_k: int = 5, fusion_pool: int = 20) -> dict:
    """Full pipeline: dense + BM25 search (each retrieving fusion_pool
    candidates) -> RRF fusion -> cross-encoder rerank to top_k.

    Returns {"results": [...], "latency_ms": {...}} so per-stage timing
    is visible, not just the final answer -- needed for the latency
    breakdown the plan calls for in Step 5/system-design discussion.
    """
    latency = {}

    t0 = time.perf_counter()
    dense_results = dense_search(query, top_k=fusion_pool)
    latency["dense_search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bm25_results = bm25_search(query, top_k=fusion_pool)
    latency["bm25_search_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    latency["fusion_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    final = rerank(query, fused[:fusion_pool], top_k=top_k)
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

    return {"results": final, "latency_ms": latency}