"""
Step 5: Evaluation methodology.

Responsibility: measure retrieval quality and answer faithfulness against
the hand-built QA set, across multiple pipeline configurations (this is
where chunk size / embedding model / retrieval mode get swept and
compared -- NOT decided upfront in Steps 1-3).

To be implemented:
- retrieval_precision_recall(qa_pairs, retriever_fn, top_k)
- faithfulness_score(answer, retrieved_chunks): a minimal custom
  implementation (decompose answer into claims, check each against
  context) -- built alongside RAGAS, not instead of it, so both numbers
  can be compared and the metric itself is explainable, not a black box.
- run_ablation(configs: list[PipelineConfig]) -> comparison table
  (chunk size x embedding model x retrieval mode)

Every number here should be reported honestly -- including configs that
perform worse than expected -- per the plan's "no cherry-picking" rule.
"""
