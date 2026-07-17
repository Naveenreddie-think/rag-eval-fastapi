"""
Step 3: Hybrid retrieval + reranking.

Responsibility: fuse dense + BM25 results (e.g. reciprocal rank fusion),
then rerank the fused candidates with a cross-encoder for final top-k.

To be implemented:
- fuse(dense_results, bm25_results): reciprocal rank fusion or weighted
  score combination
- rerank(query, candidates, top_k): cross-encoder pass over query-chunk
  pairs

Verify step (per plan): pick queries where keyword matching should
clearly matter (exact terms, function names, section headers) and show
hybrid actually beats dense-only on those -- don't just assert it.

Also the place to log per-stage latency (embedding query, dense search,
BM25 search, fusion, rerank) for the p50/p99 latency breakdown mentioned
in the system-design section of the plan.
"""
