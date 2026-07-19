
"""
Step 3: Sparse (keyword) retrieval.

BM25 index over the same chunks used for dense retrieval, for exact-match
strength embeddings are weak on (function names, exact terms, IDs).

Chunk "id" here matches the same enumerate-index convention used as the
Qdrant point ID in dense.py -- both retrievers must operate over the
identical, identically-ordered chunk list for hybrid.py's fusion step to
correctly match results by id.
"""

import re

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"\w+")

_bm25: BM25Okapi | None = None
_chunks: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(chunks: list[dict]) -> None:
    """Build the BM25 index over the given chunks. Must be called with
    the exact same chunk list (same order) used to populate Qdrant."""
    global _bm25, _chunks
    _chunks = chunks
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """Return top_k chunks by BM25 score. Returns list of
    {"id", "score", "source_path", "header_path", "text"}."""
    if _bm25 is None or _chunks is None:
        raise RuntimeError("BM25 index not built -- call build_bm25_index() first")

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": i,
            "score": float(scores[i]),
            "source_path": _chunks[i]["source_path"],
            "header_path": _chunks[i]["header_path"],
            "text": _chunks[i]["text"],
        }
        for i in top_indices
    ]
