"""
Step 7: Serving -- Production FastAPI Backend for RAG System.

Exposes endpoints for:
- POST /api/query: Grounded QA with configurable retrieval mode, alpha blending, and latency profiling.
- GET /api/health: System health and component status.
- GET /api/metrics: Evaluation benchmark metrics summary.

Includes an asynchronous lock to serialize GPU inference and prevent CUDA OOM on 8GB VRAM.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.eval.faithfulness import is_refusal_response
from app.generation.generator import generate_answer
from app.retrieval.bm25 import bm25_search, build_bm25_index
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search

# Concurrency lock to serialize GPU requests
_INFERENCE_LOCK = asyncio.Lock()
CHUNKS_PATH = Path("data/processed/chunks.json")
RETRIEVAL_RESULTS_PATH = Path("data/processed/retrieval_eval_results.json")
GENERATION_RESULTS_PATH = Path("data/processed/generation_eval_results.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    if CHUNKS_PATH.exists():
        chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        build_bm25_index(chunks)
        print(f"[API Startup] Initialized BM25 index with {len(chunks)} chunks.")
    else:
        print("[API Startup WARNING] data/processed/chunks.json not found. Run ingestion first.")
    yield
    print("[API Shutdown] Releasing resources.")


app = FastAPI(
    title="FastAPI Docs RAG Evaluation Engine",
    description="Empirically evaluated RAG system serving FastAPI documentation with grounded answering and multi-mode retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User question")
    mode: Literal["dense", "bm25", "hybrid"] = Field(
        default="hybrid", description="Retrieval mode: dense, bm25, or hybrid"
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    blend_alpha: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight for Cross-Encoder score vs RRF (hybrid mode only)"
    )


class ContextChunk(BaseModel):
    id: int
    source_path: str
    header_path: str
    text: str
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_refusal: bool
    mode: str
    top_k: int
    context_used: list[ContextChunk]
    latency_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    chunks_file_present: bool
    device: str


# --- Endpoints ---

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint confirming index availability and device status."""
    chunks_present = CHUNKS_PATH.exists()
    count = 0
    if chunks_present:
        try:
            chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            count = len(chunks)
        except Exception:
            count = 0

    return HealthResponse(
        status="healthy" if chunks_present else "degraded",
        chunks_indexed=count,
        chunks_file_present=chunks_present,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
    )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG Query"])
async def query_rag(request: QueryRequest):
    """Execute grounded question-answering over the FastAPI documentation."""
    latency: dict[str, float] = {}

    async with _INFERENCE_LOCK:
        try:
            # 1. Retrieval Stage
            t0 = time.perf_counter()
            if request.mode == "dense":
                retrieved = dense_search(request.query, top_k=request.top_k)
                latency["dense_search_ms"] = (time.perf_counter() - t0) * 1000
            elif request.mode == "bm25":
                retrieved = bm25_search(request.query, top_k=request.top_k)
                latency["bm25_search_ms"] = (time.perf_counter() - t0) * 1000
            else:  # hybrid
                hybrid_out = hybrid_search(
                    request.query,
                    top_k=request.top_k,
                    fusion_pool=20,
                    blend_alpha=request.blend_alpha,
                )
                retrieved = hybrid_out["results"]
                latency.update(hybrid_out["latency_ms"])

            latency["total_retrieval_ms"] = (time.perf_counter() - t0) * 1000

            # 2. Generation Stage
            t1 = time.perf_counter()
            gen = generate_answer(request.query, retrieved)
            answer = gen["answer"]
            latency["generation_ms"] = (time.perf_counter() - t1) * 1000
            latency["total_pipeline_ms"] = (time.perf_counter() - t0) * 1000

            refusal = is_refusal_response(answer)

            formatted_context = [
                ContextChunk(
                    id=c["id"],
                    source_path=c["source_path"],
                    header_path=c["header_path"],
                    text=c["text"],
                    score=c.get("score"),
                    rerank_score=c.get("rerank_score"),
                    final_score=c.get("final_score"),
                )
                for c in retrieved
            ]

            return QueryResponse(
                query=request.query,
                answer=answer,
                is_refusal=refusal,
                mode=request.mode,
                top_k=request.top_k,
                context_used=formatted_context,
                latency_ms=latency,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference pipeline failure: {str(e)}",
            )


@app.get("/api/metrics", tags=["Evaluation Metrics"])
async def get_evaluation_metrics():
    """Retrieve pre-computed benchmark evaluation results."""
    retrieval_data = None
    generation_data = None

    if RETRIEVAL_RESULTS_PATH.exists():
        try:
            retrieval_data = json.loads(RETRIEVAL_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    if GENERATION_RESULTS_PATH.exists():
        try:
            generation_data = json.loads(GENERATION_RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "retrieval_benchmark": retrieval_data,
        "generation_benchmark_sample_count": len(generation_data) if isinstance(generation_data, list) else None,
    }


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)
