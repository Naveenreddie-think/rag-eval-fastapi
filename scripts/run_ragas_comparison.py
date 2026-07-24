"""
Step 5: RAGAS faithfulness vs. our custom faithfulness metric, run side
by side on the SAME generated answers (reuses data/processed/
generation_eval_results.json -- does not regenerate answers, only
re-fetches the retrieved context per question, which is fast/free
locally + one Qdrant call) so the comparison is apples-to-apples.

Run with: python -m scripts.run_ragas_comparison
"""

import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from app.retrieval.dense import dense_search
from app.eval.qa_dataset import load_qa_pairs
from app.eval.ragas_llm import ClaudeRagasLLM

TOP_K = 5
RESULTS_PATH = "data/processed/generation_eval_results.json"
OUTPUT_PATH = "data/processed/ragas_comparison_results.json"


def main() -> None:
    existing_results = {r["id"]: r for r in json.loads(open(RESULTS_PATH, encoding="utf-8").read())}
    qa_pairs = {p["id"]: p for p in load_qa_pairs()}

    rows = []
    for pid, r in existing_results.items():
        if r["category"] == "no_answer" or r.get("is_refusal"):
            continue
        question = qa_pairs[pid]["question"]
        retrieved = dense_search(question, top_k=TOP_K)
        rows.append({
            "id": pid,
            "question": question,
            "answer": r["answer"],
            "contexts": [c["text"] for c in retrieved],
            "custom_faithfulness_score": r["faithfulness_score"],
        })

    print(f"Scoring {len(rows)} pairs with RAGAS faithfulness...")

    dataset = Dataset.from_list([
        {"question": r["question"], "answer": r["answer"], "contexts": r["contexts"]}
        for r in rows
    ])

    faithfulness.llm = ClaudeRagasLLM()
    result = evaluate(dataset, metrics=[faithfulness])
    result_df = result.to_pandas()
    ragas_scores = result_df["faithfulness"].tolist()

    parse_failures = sum(1 for s in ragas_scores if s is None or (isinstance(s, float) and s != s))
    if parse_failures:
        print(f"\nNote: RAGAS itself failed to parse its own LLM output on "
              f"{parse_failures}/{len(ragas_scores)} pairs (returned None/NaN) -- "
              f"excluded from the averaged comparison below, not treated as 0.")

    def _is_missing(x):
        return x is None or (isinstance(x, float) and x != x)

    comparison = []
    for row, ragas_score in zip(rows, ragas_scores):
        missing = _is_missing(ragas_score)
        comparison.append({
            "id": row["id"],
            "custom_faithfulness_score": row["custom_faithfulness_score"],
            "ragas_faithfulness_score": None if missing else ragas_score,
            "diff": None if missing else abs(row["custom_faithfulness_score"] - ragas_score),
        })

    avg_custom = sum(c["custom_faithfulness_score"] for c in comparison) / len(comparison)
    valid_ragas = [c["ragas_faithfulness_score"] for c in comparison if c["ragas_faithfulness_score"] is not None]
    avg_ragas = sum(valid_ragas) / len(valid_ragas) if valid_ragas else 0.0

    print(f"\nAverage custom faithfulness: {avg_custom:.3f}")
    print(f"Average RAGAS faithfulness:  {avg_ragas:.3f}")

    biggest_diffs = sorted([c for c in comparison if c["diff"] is not None],
                            key=lambda c: c["diff"], reverse=True)[:5]
    print("\nBiggest disagreements (custom vs RAGAS):")
    for c in biggest_diffs:
        print(f"  {c['id']}: custom={c['custom_faithfulness_score']:.2f}  "
              f"ragas={c['ragas_faithfulness_score']:.2f}  diff={c['diff']:.2f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved full comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()