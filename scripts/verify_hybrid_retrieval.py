"""
Step 3 verification: compare dense-only vs. hybrid+rerank on queries
where keyword matching should plausibly help -- including query 3 from
Step 2, which came back weaker on dense-only retrieval.

Run with: python -m scripts.verify_hybrid_retrieval
"""

import json

from app.retrieval.bm25 import build_bm25_index
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import hybrid_search, _get_cross_encoder

TEST_QUERIES = [
    "How do I define a path parameter with a type hint?",
    "How does dependency injection work in FastAPI?",
    "What is the difference between async and sync path operations?",
    "How do I set a response_model in a path operation decorator?",  # exact-term-heavy, good BM25 test
]


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    print(f"Building BM25 index over {len(chunks)} chunks...")
    build_bm25_index(chunks)
    print("Done.\n")

    # Warm up the cross-encoder (triggers model download/load) before
    # timing anything, so the first real query's rerank_ms isn't
    # polluted by one-time download/load overhead.
    print("Warming up cross-encoder...")
    _get_cross_encoder()
    print("Done.\n")

    for query in TEST_QUERIES:
        print("=" * 80)
        print("QUERY:", query)

        print("\n-- Dense-only --")
        for r in dense_search(query, top_k=3):
            print(f"  [{r['score']:.3f}] {r['source_path']} > {r['header_path']}")

        print("\n-- Hybrid + reranked --")
        out = hybrid_search(query, top_k=3)
        for r in out["results"]:
            print(f"  [rerank={r['rerank_score']:.3f}, rrf={r['rrf_score']:.4f}] "
                  f"{r['source_path']} > {r['header_path']}")
        print(f"\n  latency: {out['latency_ms']}")
        print()


if __name__ == "__main__":
    main()