"""
Step 5: Custom faithfulness/groundedness metric.

Built alongside RAGAS (not instead of it) so the metric is explainable
rather than a black box: decompose the generated answer into atomic
factual claims, then check each claim against the retrieved context.
faithfulness_score = (# claims supported by context) / (# total claims),
computed in code from real per-claim judgments -- never asking the model
to self-report the aggregate score directly.
"""

import re

from app.generation.local_llm import generate as llm_generate

CORE_REFUSAL_MARKER = "don't have enough information in the provided context"


def is_refusal_response(answer: str) -> bool:
    """Direct check for whether the model actually refused, independent
    of claim decomposition.

    Matches a stable CORE phrase fragment rather than the full exact
    sentence -- Qwen2.5-3B was observed paraphrasing slightly
    ("...to answer this question." instead of the exact "...to answer
    this."), which is obviously still a refusal but failed a strict
    exact-substring check, causing 5 genuine refusals in the no_answer
    eval to be misclassified as hallucinations. This was a bug in our
    own measurement, not a model failure."""
    return CORE_REFUSAL_MARKER in answer.lower()

def decompose_claims(answer: str) -> list[str]:
    """Break an answer into a list of atomic, independently-checkable
    factual claims. Returns a plain list of claim strings.

    IMPORTANT: only call this on answers already confirmed NOT to be a
    refusal (see custom_faithfulness_score, which checks
    is_refusal_response() first). Earlier versions asked the model to
    also decide "is this a refusal, respond NONE" as part of this same
    call -- that was unreliable and got LESS stable with more
    disambiguating examples added. Since we already have a perfectly
    reliable, deterministic string-match check for refusals, there's no
    reason to also ask an unreliable smaller model to redundantly
    re-decide the same binary question."""
    prompt = f"""Break the following answer into atomic, independently checkable \
factual claims, one claim per line, with no numbering or bullets. Each line \
must be a complete, grammatically standalone sentence -- do not split a \
single sentence across multiple lines.

Example (for format only, not the real topic below):
Answer: "Bread is made by mixing flour and water, then baking it in an oven."
Output:
Bread is made by mixing flour and water.
Bread is baked in an oven.

Now do the same for this actual answer:

Answer:
\"\"\"{answer}\"\"\"

Output (one complete claim per line):
"""
    raw_text = llm_generate(None, prompt, max_new_tokens=500)
    return _parse_claim_lines(raw_text)


def _parse_claim_lines(raw_text: str) -> list[str]:
    """Parse the line-based claim output: split on newlines, strip
    leading bullets/numbering, drop empty lines, and MERGE consecutive
    lines where the current line doesn't end in terminal punctuation
    (.!?) -- a defense against the model soft-wrapping a single
    sentence across two physical output lines."""
    stripped = raw_text.strip()
    if not stripped:
        return []

    raw_lines = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line:
            raw_lines.append(line)

    merged = []
    buffer = ""
    for line in raw_lines:
        buffer = f"{buffer} {line}".strip() if buffer else line
        if buffer.endswith((".", "!", "?")) or buffer.rstrip().endswith(("`", '"')):
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)

    return merged


def check_claims_support(claims: list[str], context_chunks: list[dict]) -> list[dict]:
    """Check each claim against the retrieved context, ONE CLAIM PER CALL
    (not batched)."""
    if not claims:
        return []

    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)

    results = []
    for claim in claims:
        prompt = f"""Context:
\"\"\"{context_text}\"\"\"

Claim: "{claim}"

Is this claim directly supported by the context above? Be strict: the \
claim must actually follow from the context, not just be plausible or \
generally true.

Respond with ONLY the single word true or false, no other text.
"""
        raw_text = llm_generate(None, prompt, max_new_tokens=10).strip().lower()
        if "true" in raw_text and "false" not in raw_text:
            supported = True
        elif "false" in raw_text:
            supported = False
        else:
            raise ValueError(
                f"Could not parse true/false from model response for claim "
                f"{claim!r}. Raw response: {raw_text!r}"
            )
        results.append({"claim": claim, "supported": supported})

    return results


def custom_faithfulness_score(answer: str, context_chunks: list[dict]) -> dict:
    """Full pipeline: check for refusal via reliable direct string match
    FIRST (no model call needed), then decompose answer into claims,
    check each against context, compute score = supported / total in
    code."""
    if is_refusal_response(answer):
        return {"score": None, "claims": [], "is_refusal": True}

    claims = decompose_claims(answer)
    if not claims:
        return {"score": None, "claims": [], "is_refusal": True}

    checked = check_claims_support(claims, context_chunks)
    supported_count = sum(1 for c in checked if c["supported"])
    score = supported_count / len(checked)
    return {"score": score, "claims": checked, "is_refusal": False}