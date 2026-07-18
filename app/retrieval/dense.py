"""
Step 2/3: Dense retrieval.

Stands up the vector store (Qdrant Cloud, free tier) and provides
dense semantic search over embedded chunks.

Collection uses cosine distance since BGE-M3 embeddings are
L2-normalized (embed() calls normalize_embeddings=True) -- cosine
similarity on normalized vectors is equivalent to dot product, and
Qdrant's COSINE distance metric handles this directly.
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.embeddings.embedder import embed, embed_query

load_dotenv()

COLLECTION_NAME = "fastapi_docs"
VECTOR_SIZE = 1024  # BGE-M3 dense embedding dimension

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
    return _client


def create_collection(recreate: bool = False) -> None:
    """Create the Qdrant collection. If recreate=True, drops and
    recreates it (useful when re-ingesting after a chunking change)."""
    client = _get_client()
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[dict]) -> None:
    """Embed and index a list of chunk dicts (each with source_path,
    header_path, text) into Qdrant. Chunk index in the list becomes its
    point ID."""
    client = _get_client()
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    points = [
        PointStruct(
            id=i,
            vector=vector,
            payload={
                "source_path": chunk["source_path"],
                "header_path": chunk["header_path"],
                "text": chunk["text"],
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def dense_search(query: str, top_k: int = 5) -> list[dict]:
    """Embed the query and return the top_k nearest chunks by cosine
    similarity. Returns list of {"score", "source_path", "header_path", "text"}."""
    client = _get_client()
    query_vector = embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points

    return [
        {
            "score": r.score,
            "source_path": r.payload["source_path"],
            "header_path": r.payload["header_path"],
            "text": r.payload["text"],
        }
        for r in results
    ]