"""
Diagnostic for Step 5: WHY does hybrid+rerank underperform dense-only on
recall? Hypothesis: reciprocal_rank_fusion() ranks ALL fused candidates,
but hybrid_search() truncates to fusion_pool=20 BEFORE handing candidates
to the reranker -- so a chunk that ranks well in dense-only but doesn't
appear in BM25's top-20 at all may get pushed out of the fused top-20
purely by RRF arithmetic, before the reranker ever gets a chance to see
and correctly re-score it.

For every single_hop/multi_hop query where dense-only found a ground
truth chunk in its top-5 but hybrid did NOT, this script shows:
- Did the ground truth chunk appear in the FULL (untruncated) RRF fusion
  ranking at all?
- If so, at what rank -- inside or outside the fusion_pool=20 cutoff?

Run with: python -m scripts.diagnose_hybrid_regression
"""

import json

from app.retrieval.bm25 import build_bm25_index, bm25_search
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import reciprocal_rank_fusion, rerank
from app.eval.qa_dataset import load_qa_pairs, resolve_ground_truth_ids

TOP_K = 5
FUSION_POOL = 20


def main() -> None:
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    build_bm25_index(chunks)

    qa_pairs = load_qa_pairs()
    resolved = resolve_ground_truth_ids(qa_pairs)
    resolved = [p for p in resolved if p["category"] in ("single_hop", "multi_hop")]

    cutoff_truncation_count = 0
    reranker_demotion_count = 0
    not_in_fusion_at_all_count = 0
    total_regressions = 0

    for pair in resolved:
        question = pair["question"]
        gt_ids = set(pair["ground_truth_ids"])

        dense_results = dense_search(question, top_k=FUSION_POOL)
        dense_top5_ids = [r["id"] for r in dense_results[:TOP_K]]
        dense_hit = bool(gt_ids & set(dense_top5_ids))

        if not dense_hit:
            continue  # only care about cases dense-only got right

        bm25_results = bm25_search(question, top_k=FUSION_POOL)
        fused_full = reciprocal_rank_fusion([dense_results, bm25_results])
        fused_full_ids = [r["id"] for r in fused_full]

        hybrid_top5 = rerank(question, fused_full[:FUSION_POOL], top_k=TOP_K)
        hybrid_top5_ids = [r["id"] for r in hybrid_top5]
        hybrid_hit = bool(gt_ids & set(hybrid_top5_ids))

        if hybrid_hit:
            continue  # not a regression case

        total_regressions += 1
        print(f"\n--- REGRESSION: {pair['id']} ---")
        print(f"    Q: {question}")

        for gt_id in gt_ids:
            if gt_id in fused_full_ids:
                rank = fused_full_ids.index(gt_id) + 1
                if rank <= FUSION_POOL:
                    reranker_demotion_count += 1
                    print(f"    GT chunk {gt_id}: rank {rank} in full RRF fusion "
                          f"(INSIDE pool={FUSION_POOL}) -- reranker demoted it")
                else:
                    cutoff_truncation_count += 1
                    print(f"    GT chunk {gt_id}: rank {rank} in full RRF fusion "
                          f"(OUTSIDE pool={FUSION_POOL}) -- truncated before rerank")
            else:
                not_in_fusion_at_all_count += 1
                print(f"    GT chunk {gt_id}: NOT in fused list at all "
                      f"(missed by both dense and BM25's top-{FUSION_POOL})")

    print(f"\n{'=' * 60}")
    print(f"Total regression cases (dense hit, hybrid miss): {total_regressions}")
    print(f"  - Cut off by fusion_pool truncation before rerank: {cutoff_truncation_count}")
    print(f"  - Demoted BY the reranker despite being in the pool: {reranker_demotion_count}")
    print(f"  - Missing from BOTH dense and BM25 top-{FUSION_POOL} entirely: {not_in_fusion_at_all_count}")


if __name__ == "__main__":
    main()