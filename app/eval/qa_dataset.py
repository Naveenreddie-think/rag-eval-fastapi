# app/eval/qa_dataset.py
"""
Step 4/5: Hand-built evaluation set loading + ground-truth resolution.

Loads eval/qa_pairs.json and resolves each pair's source_chunks
references (e.g. "tutorial/path-params.md#path-parameters-with-types")
into actual chunk indices against data/processed/chunks.json, using the
real anchor captured during chunking (see app/ingestion/chunker.py) --
not a guessed slugification, which was verified to be unreliable for
headers containing embedded HTML/markdown formatting.
"""

import json
from pathlib import Path

QA_PATH = Path("eval/qa_pairs.json")
CHUNKS_PATH = Path("data/processed/chunks.json")


def load_qa_pairs() -> list[dict]:
    """Load the hand-built QA pairs from eval/qa_pairs.json."""
    data = json.loads(QA_PATH.read_text(encoding="utf-8"))
    return data["qa_pairs"]


def _load_chunks() -> list[dict]:
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def build_anchor_lookup(chunks: list[dict]) -> dict[tuple[str, str], int]:
    """Build a (source_path, anchor) -> chunk_index lookup table."""
    lookup = {}
    for i, c in enumerate(chunks):
        if c.get("anchor"):
            lookup[(c["source_path"], c["anchor"])] = i
    return lookup


def resolve_ground_truth_ids(qa_pairs: list[dict], chunks: list[dict] | None = None) -> list[dict]:
    """Attach a 'ground_truth_ids' field (list[int]) to each QA pair by
    resolving its source_chunks references against real chunk anchors.
    no_answer pairs get an empty list (they have no source_chunks by
    design). Raises if any single_hop/multi_hop reference fails to
    resolve -- a silent resolution failure would corrupt retrieval
    metrics without any visible error."""
    if chunks is None:
        chunks = _load_chunks()
    lookup = build_anchor_lookup(chunks)

    resolved_pairs = []
    for pair in qa_pairs:
        ground_truth_ids = []
        for ref in pair["source_chunks"]:
            path, anchor = ref.split("#", 1) if "#" in ref else (ref, None)
            key = (path, anchor)
            if key not in lookup:
                raise ValueError(
                    f"QA pair {pair['id']!r} references unresolved chunk {ref!r} "
                    f"-- check chunker anchor capture or the QA pair's reference"
                )
            ground_truth_ids.append(lookup[key])
        resolved_pairs.append({**pair, "ground_truth_ids": ground_truth_ids})
    return resolved_pairs


def validate_qa_pairs(qa_pairs: list[dict]) -> None:
    """Sanity checks: non-empty question/answer, valid category, no
    duplicate IDs, no duplicate questions, and category-appropriate
    source_chunks (empty for no_answer, non-empty otherwise)."""
    valid_categories = {"single_hop", "multi_hop", "no_answer"}
    seen_ids = set()
    seen_questions = set()

    for pair in qa_pairs:
        assert pair["question"].strip(), f"{pair['id']}: empty question"
        assert pair["answer"].strip(), f"{pair['id']}: empty answer"
        assert pair["category"] in valid_categories, f"{pair['id']}: invalid category {pair['category']!r}"

        assert pair["id"] not in seen_ids, f"duplicate id: {pair['id']}"
        seen_ids.add(pair["id"])

        q_normalized = pair["question"].strip().lower()
        assert q_normalized not in seen_questions, f"{pair['id']}: duplicate question text"
        seen_questions.add(q_normalized)

        if pair["category"] == "no_answer":
            assert not pair["source_chunks"], f"{pair['id']}: no_answer pair should have empty source_chunks"
        else:
            assert pair["source_chunks"], f"{pair['id']}: {pair['category']} pair has empty source_chunks"