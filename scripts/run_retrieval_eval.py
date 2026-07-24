"""
Step 5 entry point: retrieval precision/recall/MRR against dense-only,
pure-rerank hybrid (the old, diagnosed-as-regressing behavior), and
blended hybrid at a few alpha values -- so the fix is chosen based on
real measured numbers, not assumed.

Run with: python -m scripts.run_retrieval_eval
"""

import json

from app.retrieval.bm25 import build_bm25_index
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search
from app.eval.metrics import retrieval_precision_recall

TOP_K = 5
ALPHAS_TO_TEST = [1.0, 0.7, 0.5, 0.3]  # 1.0 = old pure-reranker behavior


def dense_retriever(question: str, top_k: int) -> list[int]:
    results = dense_search(question, top_k=top_k)
    return [r["id"] for r in results]


def make_hybrid_retriever(alpha: float):
    def _retriever(question: str, top_k: int) -> list[int]:
        out = hybrid_search(question, top_k=top_k, blend_alpha=alpha)
        return [r["id"] for r in out["results"]]
    return _retriever


def print_report(name: str, result: dict) -> None:
    print(f"\n{'=' * 60}\n{name} (top_k={result['top_k']})\n{'=' * 60}")
    for cat, m in result["by_category"].items():
        print(f"  {cat:12s} (n={m['n']:2d})  "
              f"hit_rate={m['hit_rate']:.3f}  recall={m['recall']:.3f}  "
              f"precision={m['precision']:.3f}  mrr={m['mrr']:.3f}")
    o = result["overall"]
    print(f"  {'OVERALL':12s} (n={o['n']:2d})  "
          f"hit_rate={o['hit_rate']:.3f}  recall={o['recall']:.3f}  "
          f"precision={o['precision']:.3f}  mrr={o['mrr']:.3f}")


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    print(f"Building BM25 index over {len(chunks)} chunks...")
    build_bm25_index(chunks)
    print("Done.")

    all_results = {}

    print("\nRunning dense-only retrieval eval...")
    dense_result = retrieval_precision_recall(dense_retriever, top_k=TOP_K)
    print_report("DENSE-ONLY", dense_result)
    all_results["dense_only"] = dense_result

    for alpha in ALPHAS_TO_TEST:
        label = f"HYBRID (blend_alpha={alpha})" + (" -- OLD pure-rerank behavior" if alpha == 1.0 else "")
        print(f"\nRunning {label}...")
        result = retrieval_precision_recall(make_hybrid_retriever(alpha), top_k=TOP_K)
        print_report(label, result)
        all_results[f"hybrid_alpha_{alpha}"] = result

    with open("data/processed/retrieval_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved full results to data/processed/retrieval_eval_results.json")


if __name__ == "__main__":
    main()