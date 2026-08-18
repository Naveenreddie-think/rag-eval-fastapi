"""
Step 5: Multi-Dimensional Ablation Sweep.

Evaluates and compares:
1. Chunking Strategy: Structure-aware (Headers) vs. Fixed-Size (500 words) vs. Fixed-Size (200 words).
2. Embedding Models: BAAI/bge-m3 (1024d) vs. sentence-transformers/all-MiniLM-L6-v2 (384d).
3. Retrieval Modes: Dense-Only vs. BM25-Only vs. Hybrid (Blended alpha=0.7).

Metrics: Hit Rate@5, Recall@5, Precision@5, MRR across 65 ground-truth QA pairs.

Run with:
    python -m scripts.run_ablation_sweep
"""

import json
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.eval.metrics import _query_metrics
from app.eval.qa_dataset import load_qa_pairs, resolve_ground_truth_ids, validate_qa_pairs
from app.ingestion.chunker import chunk_by_headers, chunk_fixed_size
from app.ingestion.loader import load_local_docs
from app.retrieval.bm25 import bm25_search, build_bm25_index
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search

TOP_K = 5
OUTPUT_PATH = Path("data/processed/full_ablation_matrix.json")


def evaluate_in_memory_dense(chunks: list[dict], model_name: str, qa_pairs: list[dict]) -> dict:
    """Evaluate in-memory dense retrieval for a given chunk list and embedding model."""
    print(f"  Encoding {len(chunks)} chunks with {model_name}...")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    
    texts = [c["text"] for c in chunks]
    chunk_embeddings = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    per_query = []
    for pair in qa_pairs:
        q_emb = model.encode([pair["question"]], normalize_embeddings=True)[0]
        sims = np.dot(chunk_embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:TOP_K].tolist()
        
        gt_ids = pair.get("ground_truth_ids", [])
        m = _query_metrics(gt_ids, top_indices, TOP_K)
        per_query.append({"id": pair["id"], "category": pair["category"], **m})

    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
        "n": len(per_query),
    }


def evaluate_fixed_size_bm25(docs: list[dict], qa_pairs: list[dict], chunk_size: int, overlap: int) -> dict:
    """Evaluate BM25 over fixed-size chunks by source-path matching against ground truth source files."""
    fixed_chunks = []
    for d in docs:
        fixed_chunks.extend(chunk_fixed_size(d["text"], d["source_path"], size=chunk_size, overlap=overlap))

    build_bm25_index(fixed_chunks)

    per_query = []
    for pair in qa_pairs:
        results = bm25_search(pair["question"], top_k=TOP_K)
        retrieved_sources = [r["source_path"] for r in results]
        
        # Ground truth source paths
        gt_sources = set()
        for chunk_ref in pair["source_chunks"]:
            src = chunk_ref.split("#")[0]
            gt_sources.add(src)

        hits = [s for s in retrieved_sources if s in gt_sources]
        hit = 1.0 if hits else 0.0
        recall = len(set(hits)) / len(gt_sources) if gt_sources else 0.0
        precision = len(hits) / TOP_K
        mrr = 0.0
        for rank, s in enumerate(retrieved_sources, start=1):
            if s in gt_sources:
                mrr = 1.0 / rank
                break

        per_query.append({"id": pair["id"], "hit": hit, "recall": recall, "precision": precision, "mrr": mrr})

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
        "n": len(per_query),
        "total_chunks": len(fixed_chunks),
    }


def main() -> None:
    print("=" * 85)
    print("STARTING FULL MULTI-DIMENSIONAL ABLATION SWEEP")
    print("=" * 85)

    docs = load_local_docs()
    raw_qa = load_qa_pairs()
    validate_qa_pairs(raw_qa)

    chunks_path = Path("data/processed/chunks.json")
    header_chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []
    if not header_chunks:
        for d in docs:
            header_chunks.extend(chunk_by_headers(d["text"], d["source_path"]))

    resolved_qa = [p for p in resolve_ground_truth_ids(raw_qa, header_chunks) if p["category"] in ("single_hop", "multi_hop")]
    print(f"Evaluated on {len(resolved_qa)} ground-truth QA pairs across {len(header_chunks)} header chunks.\n")

    matrix = {}

    # 1. Baseline: Header Chunks + BGE-M3 (Qdrant Cloud Dense)
    print("[1/6] Evaluating Header-Aware Chunks + BGE-M3 (Qdrant Dense)...")
    from scripts.run_eval import dense_retriever
    from app.eval.metrics import retrieval_precision_recall
    d_res = retrieval_precision_recall(dense_retriever, top_k=TOP_K)
    matrix["Header-Aware | BGE-M3 | Dense-Only"] = d_res["overall"]

    # 2. Header Chunks + BM25-Only
    print("[2/6] Evaluating Header-Aware Chunks + BM25-Only...")
    build_bm25_index(header_chunks)
    from scripts.run_eval import bm25_retriever
    bm_res = retrieval_precision_recall(bm25_retriever, top_k=TOP_K)
    matrix["Header-Aware | BM25-Only"] = bm_res["overall"]

    # 3. Header Chunks + BGE-M3 + Hybrid (alpha=0.7)
    print("[3/6] Evaluating Header-Aware Chunks + BGE-M3 | Hybrid (alpha=0.7)...")
    from scripts.run_eval import make_hybrid_retriever
    hyb_res = retrieval_precision_recall(make_hybrid_retriever(0.7), top_k=TOP_K)
    matrix["Header-Aware | BGE-M3 | Hybrid (alpha=0.7)"] = hyb_res["overall"]

    # 4. Header Chunks + all-MiniLM-L6-v2 (Embedding Model Ablation)
    print("[4/6] Evaluating Header-Aware Chunks + all-MiniLM-L6-v2 (Dense-Only)...")
    minilm_res = evaluate_in_memory_dense(header_chunks, "sentence-transformers/all-MiniLM-L6-v2", resolved_qa)
    matrix["Header-Aware | all-MiniLM-L6-v2 | Dense-Only"] = minilm_res

    # 5. Fixed-Size 500w Chunks + BM25
    print("[5/6] Evaluating Fixed-Size 500w Chunks | BM25...")
    f500_res = evaluate_fixed_size_bm25(docs, resolved_qa, chunk_size=500, overlap=50)
    matrix["Fixed-Size (500w) | BM25-Only"] = f500_res

    # 6. Fixed-Size 200w Chunks + BM25
    print("[6/6] Evaluating Fixed-Size 200w Chunks | BM25...")
    f200_res = evaluate_fixed_size_bm25(docs, resolved_qa, chunk_size=200, overlap=25)
    matrix["Fixed-Size (200w) | BM25-Only"] = f200_res

    # Restore clean BM25 index over header chunks
    build_bm25_index(header_chunks)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Configuration':<45} | {'Hit@5':<8} | {'Recall@5':<9} | {'Prec@5':<8} | {'MRR':<8}")
    print("-" * 90)
    for name, r in matrix.items():
        print(f"{name:<45} | {r['hit_rate']:<8.3f} | {r['recall']:<9.3f} | {r['precision']:<8.3f} | {r['mrr']:<8.3f}")
    print("=" * 90)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    print(f"\nFull ablation matrix saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
