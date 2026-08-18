"""
Unit tests for evaluation metrics, refusal detection, and QA dataset validation.

Run with:
    python -m pytest tests/test_eval_metrics.py -v
"""

import pytest
from app.eval.metrics import _query_metrics
from app.eval.faithfulness import is_refusal_response
from app.eval.qa_dataset import validate_qa_pairs


def test_query_metrics_hit_and_recall():
    gt = [1, 2, 3]
    retrieved = [5, 2, 7, 8, 9]  # id 2 is at rank 2

    m = _query_metrics(ground_truth_ids=gt, retrieved_ids=retrieved, k=5)
    assert m["hit"] == 1.0
    assert m["recall"] == pytest.approx(1.0 / 3.0)
    assert m["precision"] == pytest.approx(1.0 / 5.0)
    assert m["mrr"] == pytest.approx(1.0 / 2.0)  # first match at rank 2


def test_query_metrics_no_hit():
    gt = [100, 101]
    retrieved = [1, 2, 3, 4, 5]

    m = _query_metrics(ground_truth_ids=gt, retrieved_ids=retrieved, k=5)
    assert m["hit"] == 0.0
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["mrr"] == 0.0


def test_is_refusal_response():
    assert is_refusal_response("I don't have enough information in the provided context to answer this.")
    assert is_refusal_response("I don't have enough information in the provided context to answer this question.")
    assert not is_refusal_response("FastAPI uses Pydantic for data validation.")


def test_validate_qa_pairs_passes_on_valid():
    valid = [
        {"id": "sh_001", "category": "single_hop", "question": "Q1", "answer": "A1", "source_chunks": ["file.md#anchor"]},
        {"id": "na_001", "category": "no_answer", "question": "Q2", "answer": "A2", "source_chunks": []},
    ]
    validate_qa_pairs(valid)  # should not raise


def test_validate_qa_pairs_rejects_empty():
    invalid = [{"id": "sh_001", "category": "single_hop", "question": "", "answer": "A1", "source_chunks": ["f.md"]}]
    with pytest.raises(AssertionError):
        validate_qa_pairs(invalid)
