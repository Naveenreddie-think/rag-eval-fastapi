"""
Follow-up diagnostic: for specific regression cases, print the ground
truth chunk's actual text + cross-encoder score, alongside whatever the
reranker ranked ABOVE it instead -- to see qualitatively why the
cross-encoder (ms-marco-MiniLM-L-6-v2) disagreed with fusion ranking.

Run with: python -m scripts.inspect_reranker_demotion
"""

import json

from app.retrieval.bm25 import build_bm25_index, bm25_search
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import reciprocal_rank_fusion, rerank

FUSION_POOL = 20

CASES = [
    {
        "question": "What underlying library is FastAPI's `TestClient` built on?",
        "gt_ids": {495},
    },
    {
        "question": (
            "If you return a `Response` object directly from a path operation instead of a "
            "`dict`/Pydantic model, does FastAPI still validate or serialize its contents "
            "using your declared `response_model`? And relatedly, what tool would you need "
            "to use manually if you wanted to put non-JSON-serializable data (like a "
            "`datetime`) into that raw `Response`?"
        ),
        "gt_ids": {649, 650, 149},
    },
]


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    build_bm25_index(chunks)

    for case in CASES:
        question = case["question"]
        gt_ids = case["gt_ids"]

        print(f"\n{'=' * 70}\nQ: {question}\n{'=' * 70}")

        dense_results = dense_search(question, top_k=FUSION_POOL)
        bm25_results = bm25_search(question, top_k=FUSION_POOL)
        fused = reciprocal_rank_fusion([dense_results, bm25_results])

        reranked = rerank(question, fused[:FUSION_POOL], top_k=len(fused[:FUSION_POOL]))

        print("\nFull reranked order (all candidates, with scores):")
        for i, r in enumerate(reranked, start=1):
            marker = " <-- GROUND TRUTH" if r["id"] in gt_ids else ""
            print(f"  #{i:2d}  id={r['id']:4d}  rerank_score={r['rerank_score']:7.3f}  "
                  f"{r['source_path']} > {r['header_path']}{marker}")

        print("\nActual text of ground truth chunk(s):")
        for r in reranked:
            if r["id"] in gt_ids:
                print(f"\n  [id={r['id']}] {r['source_path']} > {r['header_path']}")
                print(f"  {r['text'][:400]}")

        print("\nActual text of the TOP-RANKED chunk (what beat the ground truth):")
        top = reranked[0]
        if top["id"] not in gt_ids:
            print(f"\n  [id={top['id']}] {top['source_path']} > {top['header_path']}")
            print(f"  {top['text'][:400]}")
        else:
            print("  (ground truth was actually ranked #1 here)")


if __name__ == "__main__":
    main()