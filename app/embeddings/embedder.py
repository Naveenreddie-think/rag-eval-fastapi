"""
Dense Embedding Module.

Wraps BAAI/bge-m3 dense embeddings (1024-dimensional) with L2 normalization,
providing a unified interface for embedding documents and queries.
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-m3"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME, device="cuda")
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
def unload_model() -> None:
    """Free the embedding model's GPU memory."""
    global _model
    import torch
    _model = None
    torch.cuda.empty_cache()