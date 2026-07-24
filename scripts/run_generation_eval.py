"""
Step 5 entry point: full generation + faithfulness eval across all 80
QA pairs.

Saves incrementally after every pair (not just at the end) and resumes
from an existing results file if one is present -- a prior run crashed
33/80 pairs in on a parsing bug, and losing all 33 already-paid-for API
calls to a crash at pair 34 was wasteful and avoidable.

Run with: python -m scripts.run_generation_eval
"""

import json
from pathlib import Path

from app.retrieval.bm25 import build_bm25_index
from app.retrieval.dense import dense_search
from app.generation.generator import generate_answer
from app.eval.faithfulness import custom_faithfulness_score, is_refusal_response
from app.eval.qa_dataset import load_qa_pairs, validate_qa_pairs

TOP_K = 5
RESULTS_PATH = Path("data/processed/generation_eval_results.json")


def _load_existing_results() -> dict:
    if RESULTS_PATH.exists():
        rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return {r["id"]: r for r in rows}
    return {}


def _save_results(results_by_id: dict) -> None:
    RESULTS_PATH.write_text(json.dumps(list(results_by_id.values()), indent=2), encoding="utf-8")


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    build_bm25_index(chunks)

    qa_pairs = load_qa_pairs()
    validate_qa_pairs(qa_pairs)

    results_by_id = _load_existing_results()
    if results_by_id:
        print(f"Resuming: {len(results_by_id)}/{len(qa_pairs)} pairs already completed.")

    for i, pair in enumerate(qa_pairs, start=1):
        if pair["id"] in results_by_id:
            continue

        print(f"[{i}/{len(qa_pairs)}] {pair['id']} [{pair['category']}]")
        retrieved = dense_search(pair["question"], top_k=TOP_K)
        gen = generate_answer(pair["question"], retrieved)
        answer = gen["answer"]

        row = {"id": pair["id"], "category": pair["category"], "answer": answer}

        if pair["category"] == "no_answer":
            row["correctly_refused"] = is_refusal_response(answer)
        else:
            faith = custom_faithfulness_score(answer, retrieved)
            row["faithfulness_score"] = faith["score"]
            row["is_refusal"] = faith["is_refusal"]
            row["claims"] = faith["claims"]
            if faith["is_refusal"]:
                print(f"    WARNING: {pair['id']} ({pair['category']}) was refused "
                      f"entirely -- retrieval likely failed to surface relevant context")

        results_by_id[pair["id"]] = row
        _save_results(results_by_id)

    results = list(results_by_id.values())

    for cat in ("single_hop", "multi_hop"):
        rows = [r for r in results if r["category"] == cat and not r.get("is_refusal")]
        scores = [r["faithfulness_score"] for r in rows]
        avg = sum(scores) / len(scores) if scores else 0.0
        refused = sum(1 for r in results if r["category"] == cat and r.get("is_refusal"))
        print(f"\n{cat}: avg faithfulness = {avg:.3f} (n={len(rows)}, "
              f"{refused} unexpectedly refused)")

    no_answer_rows = [r for r in results if r["category"] == "no_answer"]
    refusal_accuracy = sum(r["correctly_refused"] for r in no_answer_rows) / len(no_answer_rows)
    print(f"no_answer: refusal accuracy = {refusal_accuracy:.3f} (n={len(no_answer_rows)})")

    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()