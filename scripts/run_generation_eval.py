"""
Step 5 entry point: full generation + faithfulness eval, split into two
phases to avoid VRAM exhaustion (confirmed via Task Manager: running
retrieval and generation models simultaneously filled all 8GB VRAM).

Run with: python -m scripts.run_generation_eval
"""

import json
from pathlib import Path

from app.retrieval.dense import dense_search
from app.embeddings.embedder import unload_model
from app.generation.generator import generate_answer
from app.eval.faithfulness import custom_faithfulness_score, is_refusal_response
from app.eval.qa_dataset import load_qa_pairs, validate_qa_pairs

TOP_K = 5
RETRIEVAL_CACHE_PATH = Path("data/processed/retrieval_cache.json")
RESULTS_PATH = Path("data/processed/generation_eval_results.json")


def _load_json_dict(path: Path) -> dict:
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {r["id"]: r for r in rows} if isinstance(rows, list) else rows
    return {}


def _save_dict_as_list(path: Path, d: dict) -> None:
    path.write_text(json.dumps(list(d.values()), indent=2), encoding="utf-8")


def phase1_retrieval(qa_pairs: list[dict]) -> dict:
    cache = _load_json_dict(RETRIEVAL_CACHE_PATH)
    if cache:
        print(f"Retrieval cache: {len(cache)}/{len(qa_pairs)} pairs already cached.")

    for i, pair in enumerate(qa_pairs, start=1):
        if pair["id"] in cache:
            continue
        print(f"[retrieval {i}/{len(qa_pairs)}] {pair['id']}")
        retrieved = dense_search(pair["question"], top_k=TOP_K)
        cache[pair["id"]] = {"id": pair["id"], "retrieved": retrieved}
        _save_dict_as_list(RETRIEVAL_CACHE_PATH, cache)

    print("Phase 1 (retrieval) complete. Freeing embedding model GPU memory...")
    unload_model()
    return cache


def phase2_generation(qa_pairs: list[dict], retrieval_cache: dict) -> None:
    results_by_id = _load_json_dict(RESULTS_PATH)
    if results_by_id:
        print(f"Resuming generation: {len(results_by_id)}/{len(qa_pairs)} pairs already done.")

    for i, pair in enumerate(qa_pairs, start=1):
        if pair["id"] in results_by_id:
            continue

        print(f"[generation {i}/{len(qa_pairs)}] {pair['id']} [{pair['category']}]")
        retrieved = retrieval_cache[pair["id"]]["retrieved"]
        gen = generate_answer(pair["question"], retrieved)
        answer = gen["answer"]
        print(f"    generated answer ({len(answer)} chars)")

        row = {"id": pair["id"], "category": pair["category"], "answer": answer}

        if pair["category"] == "no_answer":
            row["correctly_refused"] = is_refusal_response(answer)
        else:
            print("    scoring faithfulness...")
            faith = custom_faithfulness_score(answer, retrieved)
            row["faithfulness_score"] = faith["score"]
            row["is_refusal"] = faith["is_refusal"]
            row["claims"] = faith["claims"]
            if faith["is_refusal"]:
                print(f"    WARNING: {pair['id']} unexpectedly refused")
            else:
                print(f"    faithfulness: {faith['score']:.2f} ({len(faith['claims'])} claims)")

        results_by_id[pair["id"]] = row
        _save_dict_as_list(RESULTS_PATH, results_by_id)

    results = list(results_by_id.values())
    for cat in ("single_hop", "multi_hop"):
        rows = [r for r in results if r["category"] == cat and not r.get("is_refusal")]
        scores = [r["faithfulness_score"] for r in rows]
        avg = sum(scores) / len(scores) if scores else 0.0
        refused = sum(1 for r in results if r["category"] == cat and r.get("is_refusal"))
        print(f"\n{cat}: avg faithfulness = {avg:.3f} (n={len(rows)}, {refused} unexpectedly refused)")

    no_answer_rows = [r for r in results if r["category"] == "no_answer"]
    refusal_accuracy = sum(r["correctly_refused"] for r in no_answer_rows) / len(no_answer_rows)
    print(f"no_answer: refusal accuracy = {refusal_accuracy:.3f} (n={len(no_answer_rows)})")
    print(f"\nSaved to {RESULTS_PATH}")


def main() -> None:
    qa_pairs = load_qa_pairs()
    validate_qa_pairs(qa_pairs)
    retrieval_cache = phase1_retrieval(qa_pairs)
    phase2_generation(qa_pairs, retrieval_cache)


if __name__ == "__main__":
    main()