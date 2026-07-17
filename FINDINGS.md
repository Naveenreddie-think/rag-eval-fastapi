# Findings

Same rigor as Project 1: every claim here should be backed by a
re-runnable script or test, and failures/limitations are reported as
prominently as successes. Nothing goes in this file until it's measured.

## Architecture \& Design Decisions

State each choice with its reasoning and rejected alternative -- not
just a components list. Fill in as decisions get made (not upfront):

* **Corpus**: FastAPI docs (Tutorial + Advanced User Guide subset).
Reason: widely known, verifiable, structurally messy enough for real
chunking decisions. Rejected: Qdrant-only, Qdrant+FastAPI combined
(see chat log for full reasoning).
* **Chunking strategy**: TBD in Step 5 (swept, not assumed).
* **Embedding model**: TBD in Step 2/5 (swept, not assumed).
* **Vector DB**: Qdrant. Reason: TBD -- fill in after Step 2.
* **Retrieval mode (dense vs hybrid vs hybrid+rerank)**: TBD in Step 5.

## Measured Results (fill in as produced)

### Retrieval precision/recall

(table goes here -- per config)

### Faithfulness / groundedness

(RAGAS score vs. custom-implemented score, per config)

### Ablation comparison

(chunk size x embedding model x retrieval mode)

### Latency

p50/p99 for full pipeline, broken down by stage (embed query, dense
search, BM25 search, fusion, rerank, generation).

### Cost model

Embedding cost (one-time + incremental) vs. per-query cost.

### Security tie-in (Step 6)

Poisoned-document test results, same format as Project 1's attack
taxonomy.

## Honest Limitations

(fill in as discovered -- don't hide these)

## Bugs Found \& Fixed

\*\*Bug 3 -- cross-platform path separator mismatch.\*\*

`load\_local\_docs()` used `str(md\_path.relative\_to(docs\_root))` to build

each chunk's `source\_path`. On Linux this produces forward slashes

(`tutorial/metadata.md`); on Windows, `pathlib` produces backslashes

(`tutorial\\metadata.md`). Tests that only checked non-emptiness or counts

passed on both platforms; tests that matched an exact source\_path string

failed only on Windows (`StopIteration` from `next()` finding no match).

Caught immediately by running the same test suite on Windows after it had

already passed on the dev machine used to build Step 1 -- exactly the

kind of platform-specific bug Project 1's `sys.executable` note existed

to pre-empt, and this one slipped through anyway. Fixed by replacing

`str(path.relative\_to(root))` with `path.relative\_to(root).as\_posix()`,

which forces forward slashes regardless of OS. Any future code that

stores or compares `source\_path` (the eval set in Step 4 especially)

depends on this being consistent -- worth remembering why `as\_posix()`

is there, not just that it is.

