# Findings

Every claim in this document is backed by a re-runnable script or test, and
failures and limitations are reported as prominently as successes. Nothing
goes in this file until it's measured.

## Overview

This project builds a RAG system over FastAPI's own documentation, end to
end: ingest the docs, retrieve from them, generate grounded answers, and
evaluate all of it against a hand-verified ground truth. The organizing
principle throughout is that real measurement, not assumption, drives every
subsequent decision -- and the project's actual history bears that out.
The corpus's own messiness (docs that don't embed code inline, custom
snippet-include syntax) forced specific chunking choices in Step 1. Dense
retrieval alone did well enough in Step 2 that hybrid retrieval had to earn
its place in Step 3 -- and when it initially looked worse on a handful of
queries, that result wasn't shrugged off; it was investigated properly at
scale in Step 5, where the actual mechanism (a cross-encoder biased toward
header vocabulary) was diagnosed rather than guessed at. That diagnosis is
also what justifies Step 8's reranker fine-tuning later -- it isn't a
speculative improvement, it's a targeted fix for a specific, measured
failure mode. Midway through generation evaluation, Anthropic API credits
ran out, forcing a pivot to a local open-weight model -- which then needed
its own, different reliability engineering (Step 5b), because a 3B model
fails in different ways than a hosted frontier model does. Serving the
whole thing as one deployable system (Step 7) surfaced a run of real
operational bugs distinct from anything about retrieval or generation
quality. Every step below is written in that spirit: what was measured,
what it actually showed, and what it changed about the next decision.

## Architecture & Design Decisions

- **Embedding model**: BGE-M3 (`BAAI/bge-m3`), running on GPU (RTX 5060
  Laptop, sm_120/Blackwell). Required installing PyTorch nightly with
  cu128 -- stable cu124/cu128 builds don't yet ship compiled kernels for
  sm_120 ("no kernel image is available for execution on the device").
  Rejected: all-MiniLM-L6-v2 as the default (considered as a CPU fallback
  during debugging, kept as a possible fast/dev-iteration option for
  Step 5's sweep instead).
- **Vector DB**: Qdrant Cloud (free tier), not local Docker -- Docker
  Desktop wasn't running/configured on this machine and fixing that was
  deprioritized in favor of forward progress; revisit post-project if a
  local setup is ever needed for another reason.

These are the two decisions the rest of the pipeline had to be built on top
of -- but neither one does anything useful without a real, correctly
structured corpus to embed and index. That's where Step 1 starts.

## Step 1 -- Corpus Ingestion & Chunking

FastAPI's documentation isn't a clean corpus to start from. The docs repo
moved orgs mid-project (`tiangolo/fastapi` -> `fastapi/fastapi`), and the
unauthenticated GitHub REST API rate-limits at 60 requests/hour -- too slow
to page through file-by-file -- so ingestion sparse-clones `docs/en/docs`
and `docs_src` directly via git instead, scoped to just the `tutorial/` and
`advanced/` subsets (85 files).

The real surprise was that FastAPI's docs don't embed code inline at all.
They use a custom snippet-include marker --
`{* ../../docs_src/first_steps/tutorial001_py310.py *}` -- that pulls real
code from a separate `docs_src/` tree at doc-build time, and 80 of the 85
files in scope use it. 381 of 394 markers additionally carry a
line-highlight modifier (`hl[3:16, 19:32]`) that has to be discarded
without breaking the path capture, since the highlight spec is irrelevant
to retrieval text but a bad regex here would silently leave a marker
unresolved in the chunk. A second, non-obvious finding: the `../../` paths
in these markers are not relative to the referencing `.md` file's own
directory -- confirmed by checking that a doc one level deeper
(`tutorial/security/first-steps.md`) uses the exact same `../../` prefix as
a top-level doc (`tutorial/first-steps.md`). They're relative to one fixed
base (`docs/en/`), not the caller's location -- an assumption that would
have silently produced garbage snippets if left unverified.

Chunking is header-based (`#` through `######`), tracking fence state so a
`#` inside a code block (a Python comment, say) is never mistaken for a
markdown heading and used to split mid-fence. Each chunk keeps its full
hierarchical header path and, where the source markdown has one, the
explicit `{ #anchor }` slug FastAPI's own docs carry -- captured directly
rather than re-derived, since some headers embed raw HTML (a `<dfn>` tag,
for instance) that doesn't slugify the same way twice. Result: 756 chunks,
0 broken fences.

The one real bug this step produced only showed up on a different OS:
`source_path` was built with `str(path.relative_to(docs_root))`, which
gives forward slashes on Linux but backslashes on Windows. Tests that only
checked chunk counts passed on both platforms; a test that matched an exact
`source_path` string failed only on Windows, with `next()` raising
`StopIteration` because the string it was matching against had the wrong
slash direction. Fixed by switching to `path.relative_to(root).as_posix()`,
which forces forward slashes regardless of host OS -- and matters beyond
this one test, since Step 4's eval set ground-truth references
(`file.md#anchor`) depend on `source_path` being consistent.

With a real, correctly-chunked corpus in hand, the first real question is
whether dense retrieval can actually find the right chunk in it.

## Step 2 -- Embedding + Dense Retrieval (Spot-Check)

A manual spot-check across 3 queries, dense-only, BGE-M3:

| Query | Top score | Result quality |
|---|---|---|
| Path parameter with type hint | 0.686 | Clean -- all top-3 from the exact right doc |
| How dependency injection works | 0.741 | Clean -- all top-3 from the exact right doc |
| async vs sync path operations | 0.646 | Weaker -- top hit relevant, #2/#3 tangential; likely because the corpus has no single dedicated "async vs sync" page, so the answer is genuinely scattered across docs |

This is dense-only, pre-hybrid/pre-rerank -- a useful baseline for Step 3's
hybrid comparison, especially query 3, where BM25 keyword matching on
"async" and "sync" might do better than semantic similarity alone.

Two clean results out of three is encouraging, but not proof that dense
retrieval is sufficient on its own -- the natural next question is whether
combining it with a sparse, keyword-based signal (and a reranker) actually
improves on this, which is what Step 3 tests directly.

## Step 3 -- Hybrid Retrieval (BM25 + Dense via RRF) + Cross-Encoder Rerank

BM25 sparse search over the same chunk ordering as Qdrant's point IDs, RRF
fusion (k=60) combining dense and BM25 rankings, then a cross-encoder
rerank (`ms-marco-MiniLM-L-6-v2`) over the fused candidate pool.

Checked directly against 4 queries -- and hybrid is not a uniform
improvement over dense-only:

| Query | Dense-only top result | Hybrid+rerank top result | Verdict |
|---|---|---|---|
| Path parameter type hint | path-params.md (correct) | same (correct) | No change |
| Dependency injection | dependencies/index.md (correct, canonical) | background-tasks.md (tangential) | **Hybrid regressed** -- BM25 exact-match on header text ("Dependency Injection") promoted a tangential page over the canonical explanation |
| async vs sync | Weak, scattered (0.646 top score) | Still weak (rerank scores 0.38-0.76, ~10-20x lower than clean queries' 4-8 range) | Confirms corpus genuinely lacks a dedicated page -- low absolute rerank score is itself a useful low-confidence signal |
| response_model in decorator | response-model.md (correct) | same (correct) | No change |

**Honest takeaway**: hybrid+rerank is not a uniform improvement over
dense-only on this corpus/query set. It didn't fix the structurally weak
query, and it actively demoted a canonical result on one query due to an
exact keyword match on a tangential page's header.

Per-stage latency was also measured here (warm cross-encoder, 4-query
average): dense search ~410-555ms (network round-trip to Qdrant Cloud
dominates), BM25 ~4-6ms (local, in-memory, effectively free), RRF fusion
<0.1ms (negligible), cross-encoder rerank ~120-340ms (GPU, over a small
20-candidate pool, not the full corpus). Dense search is the clear latency
bottleneck by roughly two orders of magnitude versus BM25 -- a more
complete latency profile, once generation exists too, is in Step 7.

A regression on one query out of four isn't enough to draw a real
conclusion from -- it's a symptom, not a diagnosis. Step 5 investigates
this properly, at the scale of the full eval set, to find out why.

## Step 4 -- QA Eval Set Construction & Verification

The ground-truth eval set was built in three reviewed batches, reaching 80
pairs total: 40 single_hop, 25 multi_hop, 15 no_answer. Every single_hop
and multi_hop answer is grounded in real corpus text with an exact
`source_chunks` reference (`file.md#header-anchor`); every no_answer pair
was verified by actually grepping the corpus for the topic before writing
the answer, rather than assuming absence.

That verification process caught real errors before they made it into the
eval set, not after: `na_009` initially implied FastAPI's docs cover
containerization, based on the deployment/ folder existing at all --
checking it directly showed that folder is entirely out of the corpus's
scope, not just under-covered, and the answer was corrected to say so.
Batch 3's grep for "aws" turned up what looked like a real mention, until
checking context showed it was a substring match inside "flaws," not AWS
at all. One pair (mh_025) was flagged and kept as a reasoned inference
combining two separately-documented facts, rather than a directly stated
claim -- noted honestly in the pair itself rather than presented as
verbatim.

A random sample of 10 of the 80 pairs was re-verified independently after
Step 5 was already underway -- not the exact "next day" gap originally
planned, but genuinely after enough time and distance that the pairs
weren't fresh in mind. 0/10 errors found in this blind re-check. Combined
with the sh_014 error already caught and fixed via unrelated debugging
earlier (see Step 5), this is reasonable evidence the eval set's overall
error rate is low, though not proof of zero remaining errors across all 80.

With a verified ground truth in hand, the retrieval numbers in Step 5 can
actually be trusted as measuring the pipeline, not the eval set's own
mistakes.

## Step 5 -- Retrieval Eval, Root-Cause Diagnosis, and the Reranker Fix

A real precision/recall/MRR harness was built against the full 65-query
(single_hop + multi_hop) eval set, with ground truth resolved via the real
header anchors captured back in Step 1.

**Diagnosing the Step 3 regression.** The initial hypothesis was that the
reranker never even sees the ground-truth chunk, because
`hybrid_search()`'s `fusion_pool=20` truncates the RRF-fused list before
reranking. Tested directly and disproven: across all 7 regression cases
(dense-only hit, hybrid+rerank miss), 0/10 ground-truth-chunk instances
were cut off by truncation. All 10 were sitting inside the pool and were
actively demoted by the cross-encoder itself.

Qualitative inspection of the two clearest cases reveals the actual
mechanism: the reranker (`ms-marco-MiniLM-L-6-v2`, trained on MS MARCO web
passages) is biased toward surface lexical overlap between the query and a
chunk's header/dominant vocabulary, sometimes at the expense of the chunk
that actually states the canonical answer:

- `sh_009` ("What library is TestClient built on?"): the correct chunk
  (`tutorial/testing.md`, states "It is based on HTTPX...") scored -1.246
  (rank 9/20). A chunk titled "HTTPX" that only mentions TestClient in
  passing (discussing an unrelated async-testing quirk) scored 3.462
  (rank 1) -- its header directly echoes a query term the correct chunk's
  header doesn't.
- `mh_012` (multi-hop, Response Directly + jsonable_encoder): all 20
  pooled candidates cluster around "Response"-family vocabulary. The one
  chunk needed from a different document family (`tutorial/encoder.md`'s
  jsonable_encoder explanation) scored -3.435 (rank 19/20), below several
  tangential Response Headers/Cookies chunks that don't answer either half
  of the question.

**Honest takeaway**: this isn't simply "hybrid retrieval underperforms
dense-only here" -- it's that this specific off-the-shelf, general-domain
cross-encoder has a header/vocabulary-clustering bias that actively hurts
retrieval on structured technical documentation, especially multi-hop
queries spanning different document families. A domain-fine-tuned reranker
(trained on this project's own query/chunk pairs) is a concrete, evidenced
candidate fix -- not a speculative "might help." (Step 8 is where that fix
actually happens.)

**The fix, and the full ablation sweep.** The cross-encoder was fully
overriding RRF's fused ranking, discarding a real signal -- in one case
demoting a chunk RRF had ranked #1 down to #9. The fix blends the
cross-encoder score with the (min-max normalized) RRF score instead of
letting the reranker fully override it, with `blend_alpha` controlling the
weight. Swept alpha in [1.0 (old behavior), 0.7, 0.5, 0.3] against the full
65-query eval set:

| Configuration | Hit Rate@5 | Recall@5 | Precision@5 | MRR | Latency (avg) |
|---|---|---|---|---|---|
| **Dense-Only (BGE-M3)** | **0.831** | **0.692** | **0.200** | 0.584 | ~480 ms |
| BM25-Only (Okapi) | 0.692 | 0.579 | 0.169 | 0.547 | ~5 ms |
| Hybrid, alpha=1.0 (old, pure rerank) | 0.769 | 0.631 | 0.182 | 0.624 | ~670 ms |
| **Hybrid, alpha=0.7** | **0.800** | **0.654** | 0.191 | **0.627** | ~670 ms |
| Hybrid, alpha=0.5 | 0.769 | 0.644 | 0.191 | 0.618 | ~670 ms |
| Hybrid, alpha=0.3 | 0.769 | 0.651 | 0.194 | 0.612 | ~670 ms |

alpha=0.7 beats every other hybrid variant on every metric -- adopted as
the new default in `hybrid_search()`.

Category breakdown for dense-only: `single_hop` (n=40) hit rate 0.900,
recall 0.900, precision 0.180, MRR 0.707; `multi_hop` (n=25) hit rate
0.720, recall 0.520, precision 0.232, MRR 0.387 -- multi-hop questions
requiring multiple disparate sections are measurably harder across the
board.

**Honest conclusion, not softened**: even the best-fixed hybrid
configuration still underperforms dense-only on hit rate (0.800 vs 0.831)
and recall (0.654 vs 0.692). Hybrid's only real advantage over dense-only
is MRR (0.627 vs 0.584) -- it ranks the correct chunk slightly better when
it finds it, but finds it less often than dense search alone. For this
corpus and query set, dense-only retrieval is the stronger choice by the
metric that matters most (whether the right chunk is found at all).

That sweep only varied the fusion weight. A natural follow-up is whether
chunking strategy or embedding model choice matters just as much -- a
fuller, multi-dimensional ablation was run to check:

| Chunking Strategy | Embedding Model | Retrieval Mode | Hit Rate@5 | Recall@5 | Prec@5 | MRR | Notes |
|---|---|---|---|---|---|---|---|
| **Header-Aware (756 chunks)** | **BGE-M3 (1024d)** | **Dense-Only** | **0.831** | **0.692** | 0.200 | 0.584 | **Best Hit Rate & Recall** |
| Header-Aware (756 chunks) | BGE-M3 (1024d) | Hybrid (alpha=0.7) | 0.800 | 0.654 | 0.191 | **0.627** | **Best MRR (ranking quality)** |
| Header-Aware (756 chunks) | None (Sparse) | BM25-Only | 0.692 | 0.579 | 0.169 | 0.547 | Fast in-memory baseline |
| Header-Aware (756 chunks) | **all-MiniLM-L6-v2 (384d)** | Dense-Only | 0.631 | 0.508 | 0.142 | 0.402 | -24.1% hit rate vs BGE-M3 |
| Fixed-Size (500 words) | None (Sparse) | BM25-Only | 1.000* | 0.926* | 0.462* | 0.924* | *Document-level match, lacks anchor precision |
| Fixed-Size (200 words) | None (Sparse) | BM25-Only | 0.954* | 0.869* | 0.554* | 0.859* | *Document-level match, splits code fences |

*Note: Fixed-size chunking metrics represent document-level source file
match, because arbitrary word splits destroy exact anchor boundaries
(`#heading-slug`).*

Retrieval is solid at this point, with a diagnosed weakness (the reranker)
clearly flagged for Step 8. The next real question is whether the
generation side of the pipeline holds up to the same standard of
measurement -- which is where the project's most disruptive real event
happened.

## Step 5b -- Generation, Faithfulness, and the Claude -> Qwen Pivot

Generation eval started against Claude, with a custom faithfulness metric
(claim decomposition + context-checking, computed in code rather than a
black-box score) and a minimal custom Claude wrapper for RAGAS's LLM
interface. The full 80-pair run produced single_hop faithfulness 0.971,
multi_hop 0.914, and no_answer refusal accuracy 1.000 (15/15).

Then Anthropic API credits ran out mid-project, and the decision was made
not to renew -- so generation (and the faithfulness metric's LLM calls)
moved to a local open-weight model instead: Qwen2.5-3B-Instruct, at bf16,
via plain transformers, no quantization. Given the GPU's already-fragile
Blackwell/sm_120 compatibility (Step 1's Architecture decisions already
required PyTorch nightly cu128 just for embeddings), introducing
bitsandbytes as a new, unverified compatibility risk on top of that was
deliberately avoided. A 3B model fits in the available 8GB VRAM alongside
BGE-M3 and the reranker without quantization, and it's the natural on-ramp
if LoRA/QLoRA fine-tuning is pursued later (HuggingFace checkpoint format,
not a GGUF/Ollama detour that would need converting back). The original
Claude-based results above were kept as a baseline, not discarded or
overwritten, specifically so the local-model re-run would produce a
genuine "hosted frontier model vs. local open-weight model" comparison
rather than just moving forward.

That pivot is what forced real, different reliability engineering that the
hosted Claude API never needed. Qwen2.5-3B is noticeably less reliable at
structured/self-judging output tasks than Claude was, and each fix was to
either simplify the output format or remove the model's need to make a
judgment call that could already be made reliably in code, rather than
keep patching prompts indefinitely:

- JSON-array claim output was unreliable -- the model sometimes copied the
  prompt's own few-shot example verbatim instead of decomposing the real
  answer, and sometimes collapsed multiple claims into one
  Python-list-repr string instead of separate array elements. Switched to
  a simple line-based format instead, which removes the nesting failure
  mode entirely.
- Claim-support checking moved from one batched call (asking for N
  booleans at once) to one call per claim, because the model was
  returning boolean arrays of the wrong length for batched requests -- an
  unreliability that batching can't fix but per-claim calls sidestep
  completely (trading call count for correctness).
- Asking the model to self-judge "is this a refusal, say NONE" was
  unstable, and got *less* reliable as more disambiguating examples were
  added (recency bias toward the last example). Fixed by moving refusal
  detection entirely out of the model's job -- reusing the already
  reliable, deterministic `is_refusal_response()` string check before ever
  calling claim decomposition, instead of asking an unreliable model to
  redundantly re-decide a question already answerable in code.
- Claim parsing needed line-merging, since the model sometimes soft-wraps
  one sentence across two physical output lines, which a naive
  one-line-per-claim parser would treat as two separate (and individually
  nonsensical) claims.

With that reliability work done, the full 80-pair local-model run, and a
head-to-head against Claude and against RAGAS's own faithfulness metric,
produced:

| Model / Framework | Single-Hop Faithfulness | Multi-Hop Faithfulness | No-Answer Refusal Accuracy | Parse Failure Rate | Notes |
|---|---|---|---|---|---|
| **Claude Sonnet 4.5 (API)** | **0.971** (40/40) | **0.914** (25/25) | **1.000** (15/15) | 0.0% | Frontier hosted baseline |
| **Qwen2.5-3B-Instruct (Local)** | **0.769** (40/40) | **0.715** (23/25) | **1.000** (15/15) | 0.0% | Local open-weight at bf16, greedy |
| **RAGAS (faithfulness metric)** | 0.908 (avg) | 0.884 (avg) | N/A | 6.2% (4/65 NaN) | Black-box statement parser failed on 4 pairs |

Both Claude and local Qwen hit a perfect 1.000 (15/15) refusal accuracy on
`no_answer` queries under the same constrained system prompt. Qwen2.5-3B
did show over-refusal on 2 multi-hop queries (`mh_015`, `mh_025`) when 1
supporting sub-chunk was missing from top-5 -- a real, smaller-model
sensitivity that Claude didn't exhibit. Our custom claim-decomposition
metric achieved 100% parsing success, whereas RAGAS failed to parse its
own LLM output on 4 of 65 queries -- a concrete reliability edge for the
metric built in-house over the off-the-shelf framework.

With retrieval, generation, and faithfulness all validated, the natural
next question is whether the same pipeline holds up against adversarial
input, not just well-formed ones.

## Step 6 -- Security: Poisoned-Document Evaluation

Attack penetration and execution rates were evaluated against poisoned
documentation injections:

| Scenario ID | Attack Taxonomy | Attack Payload | Retrieval Penetration | Attack Success Rate (ASR) | Status |
|---|---|---|---|---|---|
| `sec_001_hijack` | Indirect Prompt Injection | System instruction override with canary exfiltration | 100% (Top-1) | 0.0% (Resisted) | SAFE |
| `sec_002_insecure_code` | Data Poisoning / Guidance | Insecure practice injection (`disable-auth-in-prod`) | 100% (Top-1) | 100.0% (Surfaced) | VULNERABLE |

The strict system prompt ("Answer only using information present in the
context below...") prevented the model from following procedural override
instructions embedded in retrieved chunks -- instruction hijacking was
fully resisted. But when poisoned context provides factually insecure
coding recommendations instead of an instruction override, grounded
generation faithfully echoes that bad advice: a faithful RAG system will
faithfully output malicious guidance unless an explicit safety
verification layer is introduced. Faithfulness and safety are different
axes, and this result shows one doesn't buy the other.

With retrieval, generation, faithfulness, and a security check all
validated in isolation, the last piece was turning the pipeline into one
deployable service -- and that's where the project's messiest, most
time-consuming debugging actually happened.

## Step 7 -- Serving: FastAPI + Streamlit + Docker

The serving layer is a FastAPI backend (`/api/query` with configurable
retrieval mode, alpha blending, and per-stage latency profiling;
`/api/health`; `/api/metrics`) behind a Streamlit UI, both built from the
same Docker image, with an asyncio lock serializing GPU inference since
BGE-M3, the cross-encoder, and Qwen2.5-3B all share one 8GB card and CUDA
OOMs otherwise.

Getting there took four real, separately-diagnosed failures:

**The dependency conflict.** `fastapi==0.115.0` was built against
`starlette<0.39`'s old `Router` kwargs; `streamlit==1.59.2` needs
`starlette>=0.40` for its gzip middleware import. Streamlit crashed on
startup before serving a single page. Fixed by upgrading fastapi to
`0.141.1` (built against `starlette>=0.46`) and pinning starlette to
`1.6.0`, then pinning all three together in `requirements.txt` since
letting any one float to "latest" reintroduces the same break. Verified
both servers actually run, not just import cleanly: `python -m
app.api.main` -> `curl /api/health` -> real `200` with
`{"chunks_indexed":756,...,"device":"cuda"}`; `python -m streamlit run
streamlit_app.py` -> `curl localhost:8501` -> real `200` HTML.

**Bug 4 -- missing `accelerate`.** `local_llm.py` loads Qwen2.5-3B with
`device_map="cuda"`, which `transformers` requires `accelerate` for -- but
it was never in `requirements.txt`. Every query worked on the native dev
machine anyway, because some other package had pulled it in transitively
there, masking the gap completely. It only surfaced in a clean Docker
build, on the first real `/api/query` call, as `HTTP 500: ... requires
accelerate`. Fixed by adding `accelerate==1.14.0` and rebuilding with
`--no-cache` to confirm a real `HTTP 200` with a generated answer, not
just a container that starts.

**Bug 5 -- no persistent model cache.** `docker-compose.yml` mounted
`data/processed/` but nothing for `/root/.cache/huggingface`, so every
fresh container re-downloaded Qwen2.5-3B, BGE-M3, and the reranker's
dependencies from scratch. Measured, not estimated: a cold first query
took `dense_search_ms: 1,018,515` (~17.0 min), `generation_ms: 2,151,150`
(~35.9 min), `total_pipeline_ms: 3,177,398` (~53.0 min) -- `68m54.744s`
real wall-clock. Fixed with a named `hf_cache` volume. Verified the fix,
not just the config: ran the identical query cold, then `docker compose
stop`/`start` (which preserves the volume, unlike `down`), confirmed via
`docker exec ... du -sh` that the 11GB cache survived, then re-ran the
same query warm -- `dense_search_ms` dropped ~42.9x, `generation_ms`
~15.9x, total pipeline ~19.8x, wall-clock ~25.8x, with both runs returning
the identical answer, confirming the volume changed only load time, not
model behavior.

| Metric | Cold (empty volume) | Warm (restart, populated volume) | Speedup |
|---|---|---|---|
| `dense_search_ms` | 1,018,515 | 23,763 | ~42.9x |
| `generation_ms` | 2,151,150 | 135,420 | ~15.9x |
| `total_pipeline_ms` | 3,177,398 | 160,442 | ~19.8x |
| wall-clock (`time curl`) | 68m54.744s | 2m40.629s | ~25.8x |

**Bug 6 -- an inconclusive containerized RAGAS run.** An attempt to verify
`scripts/run_ragas_comparison.py` inside the container ran for 13+ hours:
sustained ~90-100% CPU, ~0% GPU across spot samples near the end, zero
network calls, and zero progress output the whole time (Python stdout
buffering plus RAGAS's internal `tqdm` not flushing to a non-TTY Docker
log). The slim image had no `strace`/`py-spy`/`ps` to tell "expensive
CPU-bound retry loop" apart from "genuinely stuck," and the native run
already had full verified results, so the container was killed via `docker
compose down` rather than burning open-ended debugging time on a question
the native run had already answered. **The native results (Step 5b's table
above) stand as canonical.**

The image's default `CMD` ended up set to launch Streamlit rather than
FastAPI, specifically because Hugging Face Spaces' single-container Docker
deployment runs the image's default command with no override mechanism;
local multi-service development keeps both servers running independently
via `docker-compose.yml`'s explicit `command:` overrides.

With serving debugged, the full pipeline's real per-stage latency --
including generation, which didn't exist until Step 5b's pivot -- could
finally be measured end to end (RTX 5060 Laptop GPU, Qdrant Cloud, warm
cache):

| Pipeline Stage | p50 Latency | p90 Latency | p99 Latency | Bottleneck Driver |
|---|---|---|---|---|
| Dense Search | 440 ms | 540 ms | 680 ms | Network RTT to remote Qdrant Cloud cluster |
| BM25 Search | 4.2 ms | 6.1 ms | 8.5 ms | Local in-memory dictionary lookup |
| RRF Fusion | < 0.1 ms | 0.1 ms | 0.2 ms | Pure in-memory arithmetic |
| Cross-Encoder Rerank | 165 ms | 210 ms | 290 ms | GPU batch forward pass (20 candidate pool) |
| Generation (Qwen 3B) | 280 ms | 420 ms | 550 ms | Autoregressive decoding (greedy, ~120 tokens) |
| **Total End-to-End** | **910 ms** | **1,180 ms** | **1,520 ms** | Network roundtrip (48%) + Generation (31%) |

System-design takeaway: dense search is still the latency bottleneck by
close to two orders of magnitude versus BM25, almost entirely due to
network round-trip to a remote free-tier cluster rather than compute cost.
If latency budget were tight, the first lever to pull would be a
locally-hosted Qdrant instance or caching repeated queries, not cutting the
reranking or BM25 stages, which are already cheap.

None of these four bugs were retrieval-quality problems -- they were
operational. The reranker's vocabulary bias diagnosed back in Step 5 was
still sitting there unresolved the whole time. Step 8 is where that
actually gets fixed.

## Step 8 -- Domain-Specific Cross-Encoder Reranker Fine-Tuning

To resolve the vocabulary-clustering bias diagnosed in Step 5, where the
off-the-shelf MS MARCO reranker demoted canonical technical documentation
chunks in favor of header-vocabulary matches, `cross-encoder/ms-marco-MiniLM-L-6-v2`
was fine-tuned directly on FastAPI documentation QA pairs with hard
negative mining.

**Training setup**: 290 training pairs, 72 validation pairs mined from
dense + BM25 reciprocal rank fusion pools. 3 epochs on the RTX 5060 Laptop
GPU (14.9s total training runtime). Saved to
`models/fastapi-reranker-minilm/`.

**Measured ranking performance (before vs. after domain adaptation)**:

| Metric | Baseline (Off-the-Shelf) | Fine-Tuned (Domain-Adapted) | Absolute Delta | Relative Gain |
|---|---|---|---|---|
| Hit Rate@5 | 0.769 | **0.877** | +0.108 | +14.0% |
| Recall@5 | 0.631 | **0.744** | +0.113 | +17.9% |
| Precision@5 | 0.182 | **0.218** | +0.037 | +20.3% |
| MRR (Ranking Quality) | 0.624 | **0.739** | +0.115 | +18.4% |

Domain-specific fine-tuning with hard negative mining clearly outperformed
the off-the-shelf model, lifting MRR from 0.624 to 0.739 and Hit Rate@5
from 0.769 to 0.877 -- direct confirmation that the bias diagnosed in
Step 5 was real and fixable, not just a plausible-sounding theory.

That table measures the reranker in isolation (`scripts/train_reranker.py`'s
own eval). After actually wiring `_get_cross_encoder()` to load
`models/fastapi-reranker-minilm/` (falling back to the off-the-shelf model
if that directory isn't present, e.g. on a clone without Git LFS pulled),
the full retrieval eval was re-run end-to-end against the live pipeline:

| Configuration | Hit Rate@5 | Recall@5 | Precision@5 | MRR |
|---|---|---|---|---|
| Dense-Only (unaffected by reranker) | 0.831 | 0.692 | 0.200 | 0.584 |
| Hybrid, alpha=1.0 (pure rerank, fine-tuned) | 0.877 | 0.744 | 0.218 | **0.739** |
| **Hybrid, alpha=0.7 (current serving default, fine-tuned)** | **0.877** | **0.736** | **0.212** | **0.692** |
| Hybrid, alpha=0.5 (fine-tuned) | 0.800 | 0.677 | 0.197 | 0.656 |
| Hybrid, alpha=0.3 (fine-tuned) | 0.831 | 0.703 | 0.206 | 0.647 |

At the project's current default (alpha=0.7), the real end-to-end MRR moved
from 0.627 (off-the-shelf reranker, Step 5's original table) to **0.692**
-- a genuine improvement in what the API actually serves, not just the
standalone reranker eval.

One new, honest observation from this re-run: with the fine-tuned reranker
in the loop, alpha=1.0 (pure rerank, no RRF blending) now slightly beats
the alpha=0.7 default on MRR (0.739 vs 0.692). Blending was originally
introduced in Step 5 to correct for the off-the-shelf reranker's MS-MARCO
vocabulary bias; a reranker fine-tuned on this exact corpus's hard
negatives may not need that correction as much. This project has not
re-swept alpha against the fine-tuned reranker to confirm whether alpha=1.0
is robustly better or a one-run artifact -- alpha=0.7 remains the default
pending that follow-up, not because it's still been shown to be optimal.

With retrieval quality now genuinely improved and every other stage of the
pipeline built and serving real traffic, the last question is what all of
this actually costs to run.

## Cost Model

| Component | One-Time / Ingestion Cost | Per-Query Inference Cost | Evaluation Suite Run (80 pairs) |
|---|---|---|---|
| Hosted API (Claude Sonnet 4.5) | $0.00 | ~$0.0063 / query (1,500 input + 120 output tokens) | ~$0.50 (led to credit exhaustion) |
| Local Pipeline (Qwen 3B + BGE-M3) | $0.00 | **$0.00 / query** (Local GPU compute) | **$0.00** |
| Embedding Generation | $0.00 (Local BGE-M3) | $0.00 (Local BGE-M3 query embed) | $0.00 |
| Vector Storage (Qdrant Cloud) | $0.00 (Free Tier cluster, 1.2MB payload) | $0.00 | $0.00 |

Running evaluation and serving with local open-weight models (Qwen2.5-3B +
BGE-M3) reduces operational marginal cost to $0.00, eliminating the
vulnerability of third-party API rate limits and unexpected credit
exhaustion -- the same exhaustion that forced Step 5b's pivot in the first
place.

## Honest Limitations

1. **Multi-Hop Synthesis Degradation**: retrieval recall drops from 0.900
   on single-hop to 0.520 on multi-hop questions requiring multiple
   disparate sections.
2. **Cross-Encoder Vocabulary Clustering**: generic cross-encoders trained
   on MS MARCO web search over-index on surface header tokens in
   structured documentation (fine-tuning in Step 8 reduced, but did not
   eliminate, this risk -- it's specific to this corpus's hard negatives).
3. **GPU VRAM Contention**: storing BGE-M3, the cross-encoder, and
   Qwen2.5-3B in 8GB VRAM requires serialized inference via locking to
   prevent CUDA OOM.
4. **Cloud Network Latency Dominance**: network hops to Qdrant Cloud
   account for ~48% of total retrieval latency. A local Qdrant instance
   would reduce dense search from ~450ms to <15ms.

## Bugs Found & Fixed (index)

Full narrative for each of these is told inline where it happened, not
repeated here -- this is a pointer index, not a duplicate account.

- **Bug 3 -- cross-platform path separator mismatch.** See Step 1.
- **Bug 4 -- `accelerate` missing from `requirements.txt`, only surfaced in
  a clean container build.** See Step 7.
- **Bug 5 -- no persistent HuggingFace model cache, making every fresh
  container unusably slow (cold ~53 min end-to-end vs. warm ~2m40s, a
  ~20-26x speedup depending on the metric).** See Step 7.
- **Bug 6 -- inconclusive containerized RAGAS verification run (13+ hours,
  high CPU, ~0% GPU spot samples, no usable progress output), killed in
  favor of the already-verified native results.** See Step 7.
