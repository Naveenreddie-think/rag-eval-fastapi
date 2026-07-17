"""
Step 1 entry point.

Run with:  python -m scripts.ingest
"""

import json
from pathlib import Path

from app.ingestion.loader import fetch_docs, load_local_docs
from app.ingestion.chunker import chunk_by_headers

OUTPUT_PATH = Path("data/processed/chunks.json")


def main() -> None:
    fetch_docs()
    docs = load_local_docs()
    print(f"Loaded {len(docs)} docs")

    empty_docs = [d for d in docs if not d["text"].strip()]
    assert not empty_docs, f"{len(empty_docs)} docs loaded with empty text"

    missing_snippets = sum(d["text"].count("[MISSING SNIPPET") for d in docs)
    assert missing_snippets == 0, f"{missing_snippets} snippet includes failed to resolve"

    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_by_headers(d["text"], d["source_path"]))
    print(f"Total chunks: {len(all_chunks)}")

    broken_fences = [c for c in all_chunks if c["text"].count("`" * 3) % 2 != 0]
    assert not broken_fences, f"{len(broken_fences)} chunks have an unclosed code fence"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_PATH}")
    print("\nVerification: PASSED")


if __name__ == "__main__":
    main()