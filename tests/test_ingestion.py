"""
Sanity checks for Step 1 -- run locally and pass before committing.

Run with:  python -m pytest tests/test_ingestion.py -v
"""

from app.ingestion.loader import fetch_docs, load_local_docs
from app.ingestion.chunker import chunk_by_headers, chunk_fixed_size


def _get_docs():
    fetch_docs()
    return load_local_docs()


def test_docs_load_nonempty():
    docs = _get_docs()
    assert len(docs) == 85, f"expected 85 docs (51 tutorial + 34 advanced), got {len(docs)}"
    empty = [d for d in docs if not d["text"].strip()]
    assert not empty, f"{len(empty)} docs loaded with empty text: {[d['source_path'] for d in empty]}"


def test_snippet_includes_resolved():
    docs = _get_docs()
    missing = [d["source_path"] for d in docs if "[MISSING SNIPPET" in d["text"]]
    assert not missing, f"snippet includes failed to resolve in: {missing}"
    first_steps = next(d for d in docs if d["source_path"] == "tutorial/first-steps.md")
    assert "```py" in first_steps["text"], "expected resolved python code block, snippet not inlined"
    metadata = next(d for d in docs if d["source_path"] == "tutorial/metadata.md")
    assert "{*" not in metadata["text"], "hl[] modifier snippet marker left unresolved"


def test_chunks_have_metadata():
    docs = _get_docs()
    sample = next(d for d in docs if d["source_path"] == "tutorial/metadata.md")
    chunks = chunk_by_headers(sample["text"], sample["source_path"])
    assert len(chunks) > 1
    for c in chunks:
        assert c["source_path"] == "tutorial/metadata.md"
        assert isinstance(c["header_path"], str)
        assert c["text"].strip()


def test_no_chunk_splits_a_code_block():
    docs = _get_docs()
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_by_headers(d["text"], d["source_path"]))
    broken = [c for c in all_chunks if c["text"].count("`" * 3) % 2 != 0]
    assert not broken, f"{len(broken)} chunks have an unclosed code fence"


def test_fixed_size_baseline_produces_chunks():
    docs = _get_docs()
    sample = next(d for d in docs if d["source_path"] == "tutorial/metadata.md")
    chunks = chunk_fixed_size(sample["text"], sample["source_path"], size=100, overlap=10)
    assert len(chunks) > 0
    for c in chunks:
        assert c["header_path"] == ""