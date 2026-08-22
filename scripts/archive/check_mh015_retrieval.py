"""
Check whether mh_015's necessary source chunks were actually present in
its retrieved context -- reads the saved retrieval cache, no new calls.

Run with: python -m scripts.check_mh015_retrieval
"""

import json

from app.eval.qa_dataset import load_qa_pairs, resolve_ground_truth_ids


def main() -> None:
    cache = {r["id"]: r for r in json.loads(open("data/processed/retrieval_cache.json", encoding="utf-8").read())}
    qa_pairs = load_qa_pairs()
    resolved = {p["id"]: p for p in resolve_ground_truth_ids(qa_pairs)}

    pair = resolved["mh_015"]
    retrieved = cache["mh_015"]["retrieved"]
    retrieved_ids = {r["id"] for r in retrieved}

    print("Ground truth chunk ids needed:", pair["ground_truth_ids"])
    print("Retrieved chunk ids (top-5):", sorted(retrieved_ids))
    print()
    for r in retrieved:
        print(f"  id={r['id']}  {r['source_path']} > {r['header_path']}")

    missing = set(pair["ground_truth_ids"]) - retrieved_ids
    print(f"\nGround truth chunks MISSING from retrieved context: {missing}")


if __name__ == "__main__":
    main()
