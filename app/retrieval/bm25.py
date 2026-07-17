"""
Step 3: Sparse (keyword) retrieval.

Responsibility: BM25 index over the same chunks used for dense retrieval,
for exact-match strength that embeddings are weak on (IDs, specific
terms, code identifiers, section numbers).

To be implemented:
- build_bm25_index(chunks)
- bm25_search(query, top_k)
"""
