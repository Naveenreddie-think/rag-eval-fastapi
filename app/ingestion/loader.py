"""
Step 1: Corpus loading.

Pulls the scoped FastAPI docs subset (tutorial/ + advanced/, raw markdown)
via a sparse git clone, and loads them into memory as
{"source_path": ..., "text": ...} dicts.

Real finding: FastAPI's docs do NOT embed code inline. They use a custom
snippet-include syntax -- `{* ../../docs_src/first_steps/tutorial001_py310.py *}`
-- that pulls code from a separate docs_src/ tree at doc-build time. 80 of
85 files in tutorial/+advanced/ use this pattern. load_local_docs()
resolves these markers to the actual source before chunking ever sees the
text.

Note: the repo moved orgs (tiangolo/fastapi -> fastapi/fastapi); the
unauthenticated GitHub REST API also rate-limits at 60 req/hour, so this
clones via git instead of paging through the contents API.
"""

import re
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/fastapi/fastapi.git"
REPO_ROOT = Path("data/raw/fastapi-repo")
SPARSE_PATHS = ["docs/en/docs", "docs_src"]
SUBSET_DIRS = ["tutorial", "advanced"]

# Matches FastAPI's custom snippet-include marker: {* path/to/file.py *}
# and its line-highlight variant: {* path/to/file.py hl[3:16, 19:32] *}
# 381 of 394 markers in tutorial/+advanced/ use the hl[] modifier -- the
# highlight spec is discarded here (irrelevant for retrieval text), but the
# path still has to be captured correctly or the whole marker is silently
# left unresolved in the chunk text.
SNIPPET_PATTERN = re.compile(r"\{\*\s*(\S+)[^*]*\*\}")


def fetch_docs(force: bool = False) -> None:
    """Sparse-clone the FastAPI repo's docs/ + docs_src/ trees into
    data/raw/fastapi-repo. Idempotent: skips if already present unless
    force=True, so repeated runs don't re-clone every time."""
    if REPO_ROOT.exists():
        if not force:
            return
        shutil.rmtree(REPO_ROOT)

    REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "--filter=blob:none", "--sparse",
            REPO_URL, str(REPO_ROOT),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", *SPARSE_PATHS],
        cwd=REPO_ROOT,
        check=True,
    )


def _resolve_snippets(raw_text: str, snippet_base: Path) -> str:
    """Replace {* path *} markers with the real code they reference,
    wrapped in a fenced code block matching the source file's extension.

    IMPORTANT: the '../../docs_src/...' paths in these markers are NOT
    relative to the referencing .md file's own directory -- verified by
    checking that a doc one level deeper (tutorial/security/first-steps.md)
    uses the exact same '../../' prefix as a top-level doc
    (tutorial/first-steps.md). They're relative to a fixed base:
    docs/en/ (i.e. two levels above docs/en/docs/).
    """

    def _replace(match: "re.Match[str]") -> str:
        rel_path = match.group(1)
        snippet_path = (snippet_base / rel_path).resolve()
        try:
            code = snippet_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"[MISSING SNIPPET: {rel_path}]"
        lang = snippet_path.suffix.lstrip(".") or "text"
        return f"```{lang}\n{code}\n```"

    return SNIPPET_PATTERN.sub(_replace, raw_text)


def load_local_docs() -> list[dict]:
    """Load every .md file from the tutorial/ + advanced/ subset, with
    snippet-include markers resolved to real code. Assumes fetch_docs()
    has already run. Returns a list of {"source_path": str, "text": str}.
    """
    docs_root = REPO_ROOT / "docs" / "en" / "docs"
    snippet_base = docs_root.parent  # docs/en/ -- see _resolve_snippets docstring
    results = []
    for subset in SUBSET_DIRS:
        subset_root = docs_root / subset
        for md_path in sorted(subset_root.rglob("*.md")):
            raw_text = md_path.read_text(encoding="utf-8")
            resolved_text = _resolve_snippets(raw_text, snippet_base)
            rel_source = md_path.relative_to(docs_root).as_posix()
            results.append({"source_path": rel_source, "text": resolved_text})
    return results