"""
Step 5: Unified Ablation & Evaluation Runner.

Performs a systematic ablation sweep across retrieval configurations:
- Dense-only (BGE-M3)
- BM25-only (Okapi)
- Hybrid with pure Cross-Encoder reranking (blend_alpha = 1.0)
- Hybrid with blended Cross-Encoder + RRF (blend_alpha = 0.7)
- Hybrid with balanced Cross-Encoder + RRF (blend_alpha = 0.5)

Evaluates on the hand-built ground truth QA dataset (eval/qa_pairs.json)
for hit rate, recall, precision, and MRR.

Run with:
    python -m scripts.run_eval
"""

import json
from pathlib import Path

from app.eval.metrics import retrieval_precision_recall
from app.eval.qa_dataset import load_qa_pairs, validate_qa_pairs
from app.retrieval.bm25 import bm25_search, build_bm25_index
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search

TOP_K = 5
OUTPUT_PATH = Path("data/processed/ablation_results.json")


def dense_retriever(question: str, top_k: int) -> list[int]:
    results = dense_search(question, top_k=top_k)
    return [r["id"] for r in results]


def bm25_retriever(question: str, top_k: int) -> list[int]:
    results = bm25_search(question, top_k=top_k)
    return [r["id"] for r in results]


def make_hybrid_retriever(alpha: float):
    def _retriever(question: str, top_k: int) -> list[int]:
        out = hybrid_search(question, top_k=top_k, blend_alpha=alpha)
        return [r["id"] for r in out["results"]]
    return _retriever


def format_table(configs: dict[str, dict]) -> str:
    lines = []
    lines.append(f"{'Configuration':<35} | {'Hit Rate@5':<10} | {'Recall@5':<10} | {'Precision@5':<12} | {'MRR':<8}")
    lines.append("-" * 85)
    for name, res in configs.items():
        o = res["overall"]
        lines.append(
            f"{name:<35} | {o['hit_rate']:<10.3f} | {o['recall']:<10.3f} | {o['precision']:<12.3f} | {o['mrr']:<8.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    print("=" * 85)
    print("RAG SYSTEM UNIFIED ABLATION SWEEP")
    print("=" * 85)

    # 1. Validate QA pairs
    qa_pairs = load_qa_pairs()
    validate_qa_pairs(qa_pairs)
    print(f"Loaded and validated {len(qa_pairs)} QA pairs from eval/qa_pairs.json\n")

    # 2. Ensure BM25 index is built
    chunks_path = Path("data/processed/chunks.json")
    if not chunks_path.exists():
        raise FileNotFoundError("data/processed/chunks.json not found. Run python -m scripts.ingest first.")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    build_bm25_index(chunks)
    print(f"BM25 index built over {len(chunks)} chunks.\n")

    ablation_results = {}

    # 3. Dense-only ablation
    print("[1/5] Evaluating Dense-Only (BGE-M3)...")
    ablation_results["Dense-Only (BGE-M3)"] = retrieval_precision_recall(dense_retriever, top_k=TOP_K)

    # 4. BM25-only ablation
    print("[2/5] Evaluating BM25-Only (Okapi)...")
    ablation_results["BM25-Only (Okapi)"] = retrieval_precision_recall(bm25_retriever, top_k=TOP_K)

    # 5. Hybrid pure-rerank (alpha = 1.0)
    print("[3/5] Evaluating Hybrid (alpha=1.0, pure cross-encoder rerank)...")
    ablation_results["Hybrid (alpha=1.0, pure rerank)"] = retrieval_precision_recall(
        make_hybrid_retriever(1.0), top_k=TOP_K
    )

    # 6. Hybrid blended (alpha = 0.7)
    print("[4/5] Evaluating Hybrid (alpha=0.7, blended RRF+rerank)...")
    ablation_results["Hybrid (alpha=0.7, blended)"] = retrieval_precision_recall(
        make_hybrid_retriever(0.7), top_k=TOP_K
    )

    # 7. Hybrid balanced (alpha = 0.5)
    print("[5/5] Evaluating Hybrid (alpha=0.5, balanced)...")
    ablation_results["Hybrid (alpha=0.5, balanced)"] = retrieval_precision_recall(
        make_hybrid_retriever(0.5), top_k=TOP_K
    )

    # 8. Display results
    print("\n" + "=" * 85)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 85)
    print(format_table(ablation_results))
    print("=" * 85)

    # 9. Save to disk
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")
    print(f"\nDetailed ablation metrics saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
