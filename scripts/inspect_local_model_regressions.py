"""
Diagnostic: which specific no_answer pairs got hallucinated by the local
model (vs. correctly refused), and what does mh_025's unexpected
refusal actually look like? Reads already-saved results, no new model
calls needed.

Run with: python -m scripts.inspect_local_model_regressions
"""

import json

from app.eval.qa_dataset import load_qa_pairs

RESULTS_PATH = "data/processed/generation_eval_results.json"


def main() -> None:
    results = {r["id"]: r for r in json.loads(open(RESULTS_PATH, encoding="utf-8").read())}
    qa_pairs = {p["id"]: p for p in load_qa_pairs()}

    print("=" * 70)
    print("no_answer pairs that were HALLUCINATED instead of refused:")
    print("=" * 70)
    for pid, r in results.items():
        if r["category"] == "no_answer" and not r.get("correctly_refused"):
            print(f"\n[{pid}]")
            print(f"  Q: {qa_pairs[pid]['question']}")
            print(f"  A (local model): {r['answer']}")

    print(f"\n{'=' * 70}")
    print("mh_025's unexpected refusal (should have been answerable):")
    print("=" * 70)
    r = results.get("mh_025")
    if r:
        print(f"\n  Q: {qa_pairs['mh_025']['question']}")
        print(f"  A (local model): {r['answer']}")
        print(f"  Expected answer: {qa_pairs['mh_025']['answer']}")


if __name__ == "__main__":
    main()