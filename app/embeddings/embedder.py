"""
Step 2: Embedding.

Wraps BAAI/bge-m3 (free, open-weight, MIT license) behind a single
embed(texts) -> list[vector] interface, so the rest of the pipeline
doesn't care which model is behind it -- needed for Step 5's
embedding-model sweep (a second backend gets added there, not here).

Chosen: BGE-M3, dense embeddings only (its own sparse/multi-vector modes
are intentionally unused -- BM25 sparse retrieval is built separately in
Step 3, so this isn't an oversight, it's a deliberate scope split).
Rejected: Qwen3-Embedding-8B (best MTEB score, but 8B params -- too heavy
for iterative local development on CPU). Rejected: OpenAI
text-embedding-3-small (strong and cheap, but paid -- kept as the Step 5
sweep's second backend instead of the free default).
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-m3"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=8,  # smaller batches = lower peak memory
    )
    return vectors.tolist()

def embed_query(query: str) -> list[float]:
    """Embed a single query string (same model, same normalization --
    BGE-M3 doesn't need a different instruction prefix for queries vs.
    documents, unlike e.g. e5-family models)."""
    return embed([query])[0]