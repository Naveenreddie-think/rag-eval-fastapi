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

\### Step 3: hybrid + rerank vs dense-only (4 queries)



| Query | Dense-only top result | Hybrid+rerank top result | Verdict |

|---|---|---|---|

| Path parameter type hint | path-params.md (correct) | same (correct) | No change |

| Dependency injection | dependencies/index.md (correct, canonical) | background-tasks.md (tangential) | \*\*Hybrid regressed\*\* -- BM25 exact-match on header text ("Dependency Injection") promoted a tangential page over the canonical explanation |

| async vs sync | Weak, scattered (0.646 top score) | Still weak (rerank scores 0.38-0.76, \~10-20x lower than clean queries' 4-8 range) | Confirms corpus genuinely lacks a dedicated page -- low absolute rerank score is itself a useful low-confidence signal |

| response\_model in decorator | response-model.md (correct) | same (correct) | No change |



\*\*Honest takeaway\*\*: hybrid+rerank is not a uniform improvement over

dense-only on this corpus/query set. It didn't fix the structurally weak

query, and it actively demoted a canonical result on one query due to an

exact keyword match on a tangential page's header. Worth revisiting

fusion weighting (e.g. down-weighting BM25 relative to dense, or RRF's

k constant) in Step 5 rather than assuming hybrid is strictly better.



\### Latency breakdown (per-stage, warm cross-encoder, 4-query average)



| Stage | Latency | Notes |

|---|---|---|

| Dense search | \~410-555ms | Network round-trip to Qdrant Cloud dominates -- query embedding itself is fast on GPU |

| BM25 search | \~4-6ms | Local, in-memory -- effectively free |

| RRF fusion | <0.1ms | Negligible |

| Cross-encoder rerank | \~120-340ms | Runs on GPU, over a small (20-candidate) pool, not the full corpus |



\*\*System-design takeaway\*\*: dense search is the latency bottleneck by

\~2 orders of magnitude versus BM25, almost entirely due to network

round-trip to a remote free-tier cluster rather than compute cost. If

latency budget were tight, the first lever to pull would be a

locally-hosted Qdrant instance (removing the network hop) or caching

repeated/similar queries, not cutting the reranking or BM25 stages,

which are already cheap.

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

`load\\\_local\\\_docs()` used `str(md\\\_path.relative\\\_to(docs\\\_root))` to build

each chunk's `source\\\_path`. On Linux this produces forward slashes

(`tutorial/metadata.md`); on Windows, `pathlib` produces backslashes

(`tutorial\\\\metadata.md`). Tests that only checked non-emptiness or counts

passed on both platforms; tests that matched an exact source\_path string

failed only on Windows (`StopIteration` from `next()` finding no match).

Caught immediately by running the same test suite on Windows after it had

already passed on the dev machine used to build Step 1 -- exactly the

kind of platform-specific bug Project 1's `sys.executable` note existed

to pre-empt, and this one slipped through anyway. Fixed by replacing

`str(path.relative\\\_to(root))` with `path.relative\\\_to(root).as\\\_posix()`,

which forces forward slashes regardless of OS. Any future code that

stores or compares `source\\\_path` (the eval set in Step 4 especially)

depends on this being consistent -- worth remembering why `as\\\_posix()`

is there, not just that it is.

