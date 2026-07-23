"""
Step 1: Chunking.

Provisional default (per plan): structure-aware, split on markdown
headers. NOT justified yet -- chunk size/strategy gets swept for real in
Step 5 against the eval set, not decided here. chunk_fixed_size() exists
purely to produce the "naive baseline" for that later comparison.

Each chunk keeps its full header path as metadata (e.g.
"Metadata and Docs URLs > Metadata for API"), AND the real anchor slug
of its deepest header (e.g. "metadata-for-api"), captured directly from
source rather than derived by slugifying header text. This matters:
verified that some headers contain embedded HTML/markdown formatting
(e.g. '## Data <dfn title="...">conversion</dfn> { #data-conversion }')
where a naive slugify of the visible header text would NOT reliably
reproduce the real anchor -- so Step 4's QA pairs (which reference
chunks as "file.md#anchor-slug") can only be matched exactly by
capturing the real anchor, not guessing it.

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
ANCHOR_PATTERN = re.compile(r"\{\s*#([\w-]+)\s*\}")


def _split_header_text_and_anchor(raw_header: str) -> tuple[str, str | None]:
    """Split '## Data <dfn ...>conversion</dfn> { #data-conversion }' into
    ('Data <dfn ...>conversion</dfn>', 'data-conversion'). Returns
    (cleaned_text, anchor_or_None) -- anchor is None if the header has no
    explicit { #anchor } marker (some don't)."""
    match = ANCHOR_PATTERN.search(raw_header)
    anchor = match.group(1) if match else None
    text = ANCHOR_PATTERN.sub("", raw_header).strip()
    return text, anchor


def _flush(buf: list[str], header_stack: list[tuple[int, str, "str | None"]],
           source_path: str, chunks: list[dict]) -> None:
    content = "\n".join(buf).strip()
    if content:
        deepest_anchor = header_stack[-1][2] if header_stack else None
        chunks.append({
            "source_path": source_path,
            "header_path": " > ".join(h[1] for h in header_stack),
            "anchor": deepest_anchor,
            "text": content,
        })


def chunk_by_headers(doc_text: str, source_path: str) -> list[dict]:
    """Structure-aware chunking: split on markdown headers (# - ######),
    skipping any line that looks like a header while inside a fenced code
    block. Returns a list of
    {"source_path", "header_path", "anchor", "text"} dicts.
    """
    lines = doc_text.splitlines()
    chunks: list[dict] = []
    header_stack: list[tuple[int, str, "str | None"]] = []
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
            text, anchor = _split_header_text_and_anchor(header_match.group(2))
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, text, anchor))
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
            "anchor": None,
            "text": " ".join(chunk_words),
        })
        if start + size >= len(words):
            break
    return chunks