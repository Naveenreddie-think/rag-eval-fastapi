"""
Step 3/5: Hybrid retrieval + reranking.

Fuses dense + BM25 results via Reciprocal Rank Fusion (RRF), then
reranks with a cross-encoder -- BUT blends the cross-encoder score with
the RRF score rather than letting the cross-encoder fully override it.

Why blending was added (real, evidenced bug, not a hypothetical):
diagnosing Step 5's retrieval eval showed the cross-encoder
(ms-marco-MiniLM-L-6-v2, trained on MS MARCO web-search passages)
sometimes demotes the actual answer-bearing chunk below chunks that
merely share surface vocabulary with the query's header/theme -- e.g.
demoting a chunk RRF had ranked #1 down to rank #9, in favor of a
chunk titled "HTTPX" that only mentions the query's subject in passing.
Since RRF fusion (which combines two different, real signals -- dense
semantic similarity and BM25 exact-term matching) already ranked the
correct chunk highly in that case, letting the cross-encoder fully
override it discards a real signal. Blending preserves most of the RRF
signal while still incorporating the cross-encoder's benefit where the
two agree.

RRF chosen over a weighted score combination for the FIRST fusion stage
because dense (cosine similarity, ~0-1) and BM25 (unbounded,
corpus-dependent) scores aren't on comparable scales -- RRF sidesteps
that by using rank position only. For the SECOND blend (RRF vs.
cross-encoder), both signals are min-max normalized within the
candidate pool before blending, since RRF scores and cross-encoder
logits are on very different, non-comparable scales too.
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


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rerank(query: str, candidates: list[dict], top_k: int = 5,
           blend_alpha: float | None = 0.5) -> list[dict]:
    """Cross-encoder rerank over a candidate set (NOT the whole corpus --
    too slow).

    blend_alpha controls how much weight the cross-encoder score gets
    vs. the candidates' existing rrf_score, after min-max normalizing
    both within this candidate pool:
      final_score = blend_alpha * norm_rerank + (1 - blend_alpha) * norm_rrf

    blend_alpha=1.0 reproduces the OLD pure-reranker behavior (kept
    available for comparison in Step 5's ablation, since the regression
    was diagnosed against that behavior). blend_alpha=None also means
    pure reranker behavior, for backward compatibility with any caller
    not passing this parameter.

    Requires candidates to already have an 'rrf_score' field (i.e. this
    should be called on reciprocal_rank_fusion()'s output).
    """
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = _get_cross_encoder().predict(pairs)

    if blend_alpha is None or blend_alpha >= 1.0:
        scored = list(zip(candidates, ce_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**c, "rerank_score": float(s), "final_score": float(s)} for c, s in scored[:top_k]]

    rrf_scores = [c.get("rrf_score", 0.0) for c in candidates]
    norm_rerank = _min_max_normalize(list(ce_scores))
    norm_rrf = _min_max_normalize(rrf_scores)

    blended = []
    for c, raw_ce, n_ce, n_rrf in zip(candidates, ce_scores, norm_rerank, norm_rrf):
        final = blend_alpha * n_ce + (1 - blend_alpha) * n_rrf
        blended.append({**c, "rerank_score": float(raw_ce), "final_score": float(final)})

    blended.sort(key=lambda x: x["final_score"], reverse=True)
    return blended[:top_k]


def hybrid_search(query: str, top_k: int = 5, fusion_pool: int = 20,
                   blend_alpha: float = 0.7) -> dict:    
"""Full pipeline: dense + BM25 search (each retrieving fusion_pool
    candidates) -> RRF fusion -> cross-encoder rerank, blended with RRF
    score (blend_alpha=1.0 for old pure-reranker behavior).

    Returns {"results": [...], "latency_ms": {...}} so per-stage timing
    is visible, not just the final answer.
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
    final = rerank(query, fused[:fusion_pool], top_k=top_k, blend_alpha=blend_alpha)
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000

    return {"results": final, "latency_ms": latency}