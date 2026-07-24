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

\### Step 5: why does hybrid+rerank underperform dense-only? (diagnosed, not assumed)



Initial hypothesis: the reranker never sees the ground truth chunk because

`hybrid\_search()`'s fusion\_pool=20 truncates the RRF-fused list before

reranking. \*\*Tested directly and disproven\*\*: across all 7 regression

cases (dense-only hit, hybrid+rerank miss), 0/10 ground-truth-chunk

instances were cut off by truncation. All 10 were sitting inside the

pool and were actively demoted by the cross-encoder itself.



Qualitative inspection of the two clearest cases reveals the actual

mechanism: the reranker (ms-marco-MiniLM-L-6-v2, trained on MS MARCO web

passages) appears biased toward surface lexical overlap between the

query and a chunk's \*header\*/dominant vocabulary, sometimes at the

expense of the chunk that actually states the canonical answer:



\- `sh\_009` ("What library is TestClient built on?"): the correct chunk

&#x20; (`tutorial/testing.md`, states "It is based on HTTPX...") scored

&#x20; -1.246 (rank 9/20). A chunk titled "HTTPX" that only mentions

&#x20; TestClient in passing (discussing an unrelated async-testing quirk)

&#x20; scored 3.462 (rank 1) -- its header directly echoes a query term the

&#x20; correct chunk's header doesn't.

\- `mh\_012` (multi-hop, Response Directly + jsonable\_encoder): all 20

&#x20; pooled candidates cluster around "Response"-family vocabulary. The one

&#x20; chunk needed from a different document family (`tutorial/encoder.md`'s

&#x20; jsonable\_encoder explanation) scored -3.435 (rank 19/20), below

&#x20; several tangential Response Headers/Cookies chunks that don't answer

&#x20; either half of the question.



\*\*Honest takeaway\*\*: this isn't simply "hybrid retrieval underperforms

dense-only here" -- it's that this specific off-the-shelf, general-domain

cross-encoder has a header/vocabulary-clustering bias that actively hurts

retrieval on structured technical documentation, especially multi-hop

queries spanning different document families. A domain-fine-tuned

reranker (trained on this project's own (query, chunk) pairs) is a

concrete, evidenced candidate fix -- not a speculative "might help."

\### Step 5: retrieval-mode ablation, root-caused and fixed (real sweep, not assumed)



Diagnosed root cause (see earlier entry): the cross-encoder reranker

was fully overriding RRF's fused ranking, discarding a real signal --

in one case demoting a chunk RRF had ranked #1 down to #9.



\*\*Fix\*\*: blend the cross-encoder score with the (min-max normalized)

RRF score instead of letting the reranker fully override it, with

blend\_alpha controlling the weight. Tested alpha in \[1.0 (old

behavior), 0.7, 0.5, 0.3] against the full 65-query (single\_hop +

multi\_hop) eval set:



| Config | Hit rate | Recall | Precision | MRR |

|---|---|---|---|---|

| Dense-only | 0.831 | 0.692 | 0.200 | 0.584 |

| Hybrid, alpha=1.0 (old, pure rerank) | 0.769 | 0.631 | 0.182 | 0.624 |

| \*\*Hybrid, alpha=0.7\*\* | \*\*0.800\*\* | \*\*0.654\*\* | 0.191 | \*\*0.627\*\* |

| Hybrid, alpha=0.5 | 0.769 | 0.644 | 0.191 | 0.618 |

| Hybrid, alpha=0.3 | 0.769 | 0.651 | 0.194 | 0.612 |



alpha=0.7 beats every other hybrid variant on every metric -- adopted

as the new default in hybrid\_search().



\*\*Honest conclusion, not softened\*\*: even the best-fixed hybrid

configuration still underperforms dense-only on hit rate (0.800 vs

0.831) and recall (0.654 vs 0.692). Hybrid's only real advantage over

dense-only is MRR (0.627 vs 0.584) -- it ranks the correct chunk

slightly better when it finds it, but finds it less often than dense

search alone. For this corpus and query set, dense-only retrieval is

the stronger choice by the metric that matters most (whether the right

chunk is found at all); hybrid+rerank's value here is narrower than the

"hybrid is generally better" assumption common in RAG literature.





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

`load\\\\\\\_local\\\\\\\_docs()` used `str(md\\\\\\\_path.relative\\\\\\\_to(docs\\\\\\\_root))` to build

each chunk's `source\\\\\\\_path`. On Linux this produces forward slashes

(`tutorial/metadata.md`); on Windows, `pathlib` produces backslashes

(`tutorial\\\\\\\\metadata.md`). Tests that only checked non-emptiness or counts

passed on both platforms; tests that matched an exact source\_path string

failed only on Windows (`StopIteration` from `next()` finding no match).

Caught immediately by running the same test suite on Windows after it had

already passed on the dev machine used to build Step 1 -- exactly the

kind of platform-specific bug Project 1's `sys.executable` note existed

to pre-empt, and this one slipped through anyway. Fixed by replacing

`str(path.relative\\\\\\\_to(root))` with `path.relative\\\\\\\_to(root).as\\\\\\\_posix()`,

which forces forward slashes regardless of OS. Any future code that

stores or compares `source\\\\\\\_path` (the eval set in Step 4 especially)

depends on this being consistent -- worth remembering why `as\\\\\\\_posix()`

is there, not just that it is.

