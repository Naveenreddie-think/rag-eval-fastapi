"""
Step 8: Domain-Specific Cross-Encoder Reranker Fine-Tuning.

Trains ms-marco-MiniLM-L-6-v2 directly on FastAPI documentation QA pairs:
- Positive pairs: (Question, Ground-Truth Chunk Text) -> label = 1.0
- Hard Negative pairs: (Question, High-Scoring Non-Ground-Truth Chunk Text) -> label = 0.0

Evaluates before/after retrieval ranking and fixes the vocabulary-clustering
bias diagnosed in FINDINGS.md.

Run with:
    python -m scripts.train_reranker
"""

import json
import os
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

from app.eval.metrics import _query_metrics
from app.eval.qa_dataset import load_qa_pairs, resolve_ground_truth_ids
from app.retrieval.bm25 import bm25_search, build_bm25_index
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import reciprocal_rank_fusion

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_MODEL_DIR = "models/fastapi-reranker-minilm"
RESULTS_PATH = Path("data/processed/reranker_finetune_results.json")
SEED = 42


def build_training_data(resolved_qa: list[dict], chunks: list[dict], hard_negatives_per_query: int = 4) -> tuple[list, list]:
    """Build positive and hard negative pairs from BM25/Dense candidate pools."""
    random.seed(SEED)
    train_samples = []
    val_samples = []

    # 80/20 train/val split on queries
    shuffled_qa = list(resolved_qa)
    random.shuffle(shuffled_qa)
    split_idx = int(len(shuffled_qa) * 0.8)
    train_qa = shuffled_qa[:split_idx]
    val_qa = shuffled_qa[split_idx:]

    def _create_examples(qa_subset: list[dict]) -> list[InputExample]:
        examples = []
        for pair in qa_subset:
            q = pair["question"]
            gt_ids = set(pair["ground_truth_ids"])
            
            # 1. Positive examples
            for gid in gt_ids:
                examples.append(InputExample(texts=[q, chunks[gid]["text"]], label=1.0))

            # 2. Mine hard negatives from dense + BM25 pool
            dense_cand = dense_search(q, top_k=15)
            bm25_cand = bm25_search(q, top_k=15)
            fused = reciprocal_rank_fusion([dense_cand, bm25_cand], k=60)
            
            negatives = [c for c in fused if c["id"] not in gt_ids][:hard_negatives_per_query]
            for neg in negatives:
                examples.append(InputExample(texts=[q, neg["text"]], label=0.0))
        return examples

    train_examples = _create_examples(train_qa)
    val_examples = _create_examples(val_qa)
    return train_examples, val_examples


def evaluate_reranker(model: CrossEncoder, resolved_qa: list[dict], chunks: list[dict], top_k: int = 5) -> dict:
    """Evaluate retrieval precision, recall, and MRR using the given reranker."""
    per_query = []
    for pair in resolved_qa:
        q = pair["question"]
        gt_ids = set(pair["ground_truth_ids"])

        dense_cand = dense_search(q, top_k=20)
        bm25_cand = bm25_search(q, top_k=20)
        fused = reciprocal_rank_fusion([dense_cand, bm25_cand], k=60)[:20]

        pairs = [(q, c["text"]) for c in fused]
        scores = model.predict(pairs)
        scored_candidates = sorted(zip(fused, scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_ids = [c["id"] for c, _ in scored_candidates]

        m = _query_metrics(list(gt_ids), top_ids, top_k)
        per_query.append({"id": pair["id"], **m})

    def _avg(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "hit_rate": _avg(per_query, "hit"),
        "recall": _avg(per_query, "recall"),
        "precision": _avg(per_query, "precision"),
        "mrr": _avg(per_query, "mrr"),
    }


def main() -> None:
    print("=" * 85)
    print("STEP 8: DOMAIN-SPECIFIC CROSS-ENCODER RERANKER FINE-TUNING")
    print("=" * 85)

    # 1. Load data
    chunks = json.loads(open("data/processed/chunks.json", encoding="utf-8").read())
    build_bm25_index(chunks)
    qa_pairs = load_qa_pairs()
    resolved_qa = [p for p in resolve_ground_truth_ids(qa_pairs, chunks) if p["category"] in ("single_hop", "multi_hop")]

    print(f"Loaded {len(chunks)} chunks and {len(resolved_qa)} ground-truth QA pairs.")

    # 2. Mine hard negatives & build datasets
    print("Mining hard negatives and building train/val datasets...")
    train_samples, val_samples = build_training_data(resolved_qa, chunks)
    print(f"Generated {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Baseline Evaluation (Before Fine-Tuning)
    print("\n[1/3] Evaluating Baseline Off-the-Shelf Reranker...")
    base_model = CrossEncoder(BASE_MODEL, device=device)
    baseline_metrics = evaluate_reranker(base_model, resolved_qa, chunks)
    print(f"  Baseline: Hit@5 = {baseline_metrics['hit_rate']:.3f} | Recall@5 = {baseline_metrics['recall']:.3f} | MRR = {baseline_metrics['mrr']:.3f}")

    # 4. Train / Fine-tune Reranker
    print("\n[2/3] Fine-tuning Cross-Encoder on FastAPI Technical Documentation...")
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=8)
    
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name="fastapi_val")

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    base_model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=3,
        evaluation_steps=20,
        warmup_steps=10,
        output_path=OUTPUT_MODEL_DIR,
        show_progress_bar=True,
    )
    print(f"Fine-tuned model saved to {OUTPUT_MODEL_DIR}")

    # 5. Post-Fine-Tuning Evaluation
    print("\n[3/3] Evaluating Fine-Tuned Domain-Adapted Reranker...")
    finetuned_model = CrossEncoder(OUTPUT_MODEL_DIR, device=device)
    finetuned_metrics = evaluate_reranker(finetuned_model, resolved_qa, chunks)

    # 6. Summary Comparison
    print("\n" + "=" * 85)
    print("RERANKER FINE-TUNING RESULTS (BEFORE vs. AFTER)")
    print("=" * 85)
    print(f"{'Metric':<20} | {'Baseline (Off-the-Shelf)':<25} | {'Fine-Tuned (Domain-Adapted)':<25} | {'Delta':<10}")
    print("-" * 85)
    for m_key, m_name in [("hit_rate", "Hit Rate@5"), ("recall", "Recall@5"), ("precision", "Precision@5"), ("mrr", "MRR")]:
        b = baseline_metrics[m_key]
        f = finetuned_metrics[m_key]
        delta = f - b
        print(f"{m_name:<20} | {b:<25.3f} | {f:<25.3f} | {delta:+10.3f}")
    print("=" * 85)

    comparison_payload = {
        "baseline_model": BASE_MODEL,
        "finetuned_model": OUTPUT_MODEL_DIR,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "training_samples_count": len(train_samples),
        "validation_samples_count": len(val_samples),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")
    print(f"\nSaved fine-tuning results report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
