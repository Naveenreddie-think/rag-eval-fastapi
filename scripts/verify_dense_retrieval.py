"""
Step 2 verification script.

Indexes all chunks into Qdrant and runs a handful of manual test queries
to sanity-check that dense retrieval returns semantically sensible
results, before adding BM25/hybrid on top.

Run with: python -m scripts.verify_dense_retrieval
"""

import json

from app.retrieval.dense import create_collection, upsert_chunks, dense_search

TEST_QUERIES = [
    "How do I define a path parameter with a type hint?",
    "How does dependency injection work in FastAPI?",
    "What is the difference between async and sync path operations?",
]


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    print(f"Indexing {len(chunks)} chunks...")

    create_collection(recreate=True)
    upsert_chunks(chunks)
    print("Done indexing.")

    for query in TEST_QUERIES:
        print()
        print("QUERY:", query)
        for r in dense_search(query, top_k=3):
            print(f"  [{r['score']:.3f}] {r['source_path']} > {r['header_path']}")


if __name__ == "__main__":
    main()