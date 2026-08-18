"""
Inspect mh_015's over-refusal in detail -- reads already-saved data,
no new model calls.

Run with: python -m scripts.inspect_mh015_refusal
"""

import json

from app.eval.qa_dataset import load_qa_pairs

def main() -> None:
    results = {r["id"]: r for r in json.loads(open("data/processed/generation_eval_results.json", encoding="utf-8").read())}
    qa_pairs = {p["id"]: p for p in load_qa_pairs()}

    r = results["mh_015"]
    p = qa_pairs["mh_015"]
    print("Q:", p["question"])
    print("\nExpected answer:", p["answer"])
    print("\nExpected source_chunks:", p["source_chunks"])
    print("\nLocal model's actual answer:", r["answer"])

if __name__ == "__main__":
    main()