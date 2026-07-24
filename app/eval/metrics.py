# app/eval/metrics.py
"""
Step 5: Retrieval evaluation methodology.

Measures retrieval quality against the hand-built QA set for a given
retriever function, across single_hop and multi_hop categories
(no_answer pairs have no ground truth chunks by design -- they're
evaluated separately, at the generation stage, on whether the system
honestly declines rather than hallucinates).

Metrics computed per query, then averaged per category:
- Hit Rate@k: 1 if ANY ground truth chunk appears in the top-k results, else 0
- Recall@k: (# ground truth chunks found in top-k) / (# ground truth chunks)
- Precision@k: (# ground truth chunks found in top-k) / k
- MRR: 1 / (rank of the first ground truth chunk found), 0 if none found

Multi-hop queries have >1 ground truth chunk, so Recall@k for those
specifically tests whether retrieval surfaces ALL the chunks needed to
answer, not just one -- that's the whole point of including them.
"""

from app.eval.qa_dataset import load_qa_pairs, resolve_ground_truth_ids, validate_qa_pairs


def _query_metrics(ground_truth_ids: list[int], retrieved_ids: list[int], k: int) -> dict:
    top_k_ids = retrieved_ids[:k]
    gt_set = set(ground_truth_ids)
    found = [cid for cid in top_k_ids if cid in gt_set]

    hit = 1.0 if found else 0.0
    recall = len(found) / len(gt_set) if gt_set else 0.0
    precision = len(found) / k if k else 0.0

    mrr = 0.0
    for rank, cid in enumerate(top_k_ids, start=1):
        if cid in gt_set:
            mrr = 1.0 / rank
            break

    return {"hit": hit, "recall": recall, "precision": precision, "mrr": mrr}


def retrieval_precision_recall(retriever_fn, top_k: int = 5,
                                categories: tuple[str, ...] = ("single_hop", "multi_hop")) -> dict:
    """Run retriever_fn(question, top_k) -> list of chunk ids (ordered,
    best first) against every QA pair in the given categories, and
    return per-category and overall averaged metrics, plus per-query
    detail (needed to spot individual failures, not just averages).
    """
    qa_pairs = load_qa_pairs()
    validate_qa_pairs(qa_pairs)
    resolved = resolve_ground_truth_ids(qa_pairs)
    resolved = [p for p in resolved if p["category"] in categories]

    per_query = []
    for pair in resolved:
        retrieved_ids = retriever_fn(pair["question"], top_k)
        m = _query_metrics(pair["ground_truth_ids"], retrieved_ids, top_k)
        per_query.append({"id": pair["id"], "category": pair["category"], **m})

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    result = {"top_k": top_k, "per_query": per_query, "by_category": {}, "overall": {}}
    for cat in categories:
        rows = [r for r in per_query if r["category"] == cat]
        result["by_category"][cat] = {
            "n": len(rows),
            "hit_rate": _avg(rows, "hit"),
            "recall": _avg(rows, "recall"),
            "precision": _avg(rows, "precision"),
            "mrr": _avg(rows, "mrr"),
        }
    result["overall"] = {
        "n": len(per_query),
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
    }
    return result