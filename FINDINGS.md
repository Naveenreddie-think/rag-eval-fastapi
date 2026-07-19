# Findings

Same rigor as Project 1: every claim here should be backed by a
re-runnable script or test, and failures/limitations are reported as
prominently as successes. Nothing goes in this file until it's measured.

## Architecture \& Design Decisions

* \- \*\*Embedding model\*\*: BGE-M3 (`BAAI/bge-m3`), running on GPU (RTX 5060
* &#x20; Laptop, sm\_120/Blackwell). Required installing PyTorch nightly with
* &#x20; cu128 -- stable cu124/cu128 builds don't yet ship compiled kernels for
* &#x20; sm\_120 ("no kernel image is available for execution on the device").
* &#x20; Rejected: all-MiniLM-L6-v2 as the default (considered as a CPU fallback
* &#x20; during debugging, kept as a possible fast/dev-iteration option for
* &#x20; Step 5's sweep instead).
* \- \*\*Vector DB\*\*: Qdrant Cloud (free tier), not local Docker -- Docker
* &#x20; Desktop wasn't running/configured on this machine and fixing that was
* &#x20; deprioritized in favor of forward progress; revisit post-project if a
* &#x20; local setup is ever needed for another reason.

## Measured Results (fill in as produced)

\### Step 2 manual retrieval spot-check (3 queries, dense-only, BGE-M3)



| Query | Top score | Result quality |

|---|---|---|

| Path parameter with type hint | 0.686 | Clean -- all top-3 from the exact right doc |

| How dependency injection works | 0.741 | Clean -- all top-3 from the exact right doc |

| async vs sync path operations | 0.646 | Weaker -- top hit relevant, #2/#3 tangential; likely because the corpus has no single dedicated "async vs sync" page, so the answer is genuinely scattered across docs |



This is dense-only, pre-hybrid/pre-rerank -- a useful baseline for Step 3's

hybrid comparison, especially query 3, where BM25 keyword matching on

"async" and "sync" might do better than semantic similarity alone.

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

