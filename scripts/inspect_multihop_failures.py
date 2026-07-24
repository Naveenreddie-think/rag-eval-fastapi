"""
Reads the already-saved generation_eval_results.json (no new API calls)
and prints the specific unsupported claims from multi_hop pairs that
scored below 1.0 -- concrete failure examples for FINDINGS.md, not just
the aggregate 0.914 number.

Run with: python -m scripts.inspect_multihop_failures
"""

import json

RESULTS_PATH = "data/processed/generation_eval_results.json"


def main() -> None:
    results = json.loads(open(RESULTS_PATH, encoding="utf-8").read())
    multihop = [r for r in results if r["category"] == "multi_hop" and not r.get("is_refusal")]

    below_perfect = [r for r in multihop if r["faithfulness_score"] < 1.0]
    print(f"{len(below_perfect)}/{len(multihop)} multi_hop pairs scored below 1.0\n")

    for r in sorted(below_perfect, key=lambda x: x["faithfulness_score"]):
        print(f"{'=' * 70}")
        print(f"{r['id']}  faithfulness={r['faithfulness_score']:.2f}")
        print(f"{'=' * 70}")
        for c in r["claims"]:
            if not c["supported"]:
                print(f"  [UNSUPPORTED] {c['claim']}")
        print(f"\nFull answer:\n  {r['answer']}\n")


if __name__ == "__main__":
    main()