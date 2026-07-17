"""
Step 2/3: Dense retrieval.

Responsibility: stand up the vector store (Qdrant, local Docker mode for
dev, cloud free-tier for the deployed demo) and provide dense semantic
search over embedded chunks.

To be implemented:
- upsert_chunks(chunks, embeddings): index chunks into Qdrant
- dense_search(query, top_k): embed query, return nearest chunks by
  cosine similarity

Verify step: run a handful of manual test queries, confirm returned
chunks are semantically sensible before adding BM25/hybrid on top.
"""
