"""
Step 1: Chunking.

Provisional default (per plan): structure-aware, split on markdown
headers. NOT justified yet -- chunk size/strategy gets swept for real in
Step 5 against the eval set, not decided here. chunk_fixed_size() exists
purely to produce the "naive baseline" for that later comparison.

Fence-tracking detail: chunk_by_headers() tracks whether it's inside a
fenced code block (``` ... ```) and never treats a line as a header while
inside one. Without this, a Python comment inside a code sample --
`# a heading-looking comment` -- would get misread as a markdown header
and incorrectly split the chunk mid-code-block.
"""

import re

HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_PATTERN = re.compile(r"^```")
# FastAPI's custom heading-anchor syntax, e.g. "# First Steps { #first-steps }"
ANCHOR_PATTERN = re.compile(r"\{\s*#[\w-]+\s*\}")


def _clean_header_text(text: str) -> str:
    return ANCHOR_PATTERN.sub("", text).strip()


def _flush(buf: list[str], header_stack: list[tuple[int, str]],
           source_path: str, chunks: list[dict]) -> None:
    content = "\n".join(buf).strip()
    if content:
        chunks.append({
            "source_path": source_path,
            "header_path": " > ".join(h[1] for h in header_stack),
            "text": content,
        })


def chunk_by_headers(doc_text: str, source_path: str) -> list[dict]:
    """Structure-aware chunking: split on markdown headers (# - ######),
    skipping any line that looks like a header while inside a fenced code
    block. Returns a list of {"source_path", "header_path", "text"} dicts.
    """
    lines = doc_text.splitlines()
    chunks: list[dict] = []
    header_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_fence = False

    for line in lines:
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        header_match = None if in_fence else HEADER_PATTERN.match(line)
        if header_match:
            _flush(current_lines, header_stack, source_path, chunks)
            level = len(header_match.group(1))
            text = _clean_header_text(header_match.group(2))
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, text))
            current_lines = [line]
        else:
            current_lines.append(line)

    _flush(current_lines, header_stack, source_path, chunks)
    return chunks


def chunk_fixed_size(doc_text: str, source_path: str,
                      size: int = 500, overlap: int = 50) -> list[dict]:
    """Naive baseline: fixed-size chunking by word count (word count used
    as a cheap token-count proxy -- fine for a baseline, not for the real
    pipeline). No structure awareness at all -- this is deliberately the
    'before' case, expected to split tables and code blocks mid-way."""
    words = doc_text.split()
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start:start + size]
        if not chunk_words:
            continue
        chunks.append({
            "source_path": source_path,
            "header_path": "",
            "text": " ".join(chunk_words),
        })
        if start + size >= len(words):
            break
    return chunks