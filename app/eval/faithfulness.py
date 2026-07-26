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

REFUSAL_PHRASE = "I don't have enough information in the provided context to answer this."


def is_refusal_response(answer: str) -> bool:
    """Direct check for whether the model actually refused, independent
    of claim decomposition."""
    return REFUSAL_PHRASE in answer


def decompose_claims(answer: str) -> list[str]:
    """Break an answer into a list of atomic, independently-checkable
    factual claims. Returns a plain list of claim strings.

    Uses a LINE-based format (one claim per line), not JSON. Switched
    from JSON after real, repeated failures with Qwen2.5-3B on this
    task: it sometimes copied the prompt's own few-shot example content
    verbatim instead of decomposing the real answer, and sometimes
    wrapped multiple claims into a single Python-list-repr-shaped
    string. A plain line-based format removes the JSON nesting failure
    mode entirely.

    Also fixes a second real bug: the model over-generalized "NONE" to
    mean "nothing interesting to decompose" (returning it for
    single-sentence and code-containing answers that were clearly NOT
    refusals), not just "this is a refusal" as intended -- added
    explicit examples disambiguating both cases."""
    prompt = f"""Break the following answer into atomic, independently checkable \
factual claims, one claim per line, with no numbering or bullets.

IMPORTANT: respond with exactly NONE only if the answer is purely a refusal \
or decline (e.g. states it doesn't have enough information to answer). If the \
answer provides ANY information at all -- even a single short sentence, or a \
sentence plus a code example -- extract at least one claim. A short or \
simple answer is NOT a reason to say NONE.

Example 1 (multi-sentence answer -- for format only, not the real topic):
Answer: "Bread is made by mixing flour and water, then baking it in an oven."
Output:
Bread is made by mixing flour and water.
Bread is baked in an oven.

Example 2 (single short sentence -- still extract the one claim, do NOT say NONE):
Answer: "Water boils at 100 degrees Celsius at sea level."
Output:
Water boils at 100 degrees Celsius at sea level.

Example 3 (answer includes a code example -- describe what it shows, do NOT say NONE):
Answer: "You create a list in Python using square brackets, like this: my_list = [1, 2, 3]"
Output:
You create a list in Python using square brackets.

Example 4 (genuine refusal -- this is the ONLY case that gets NONE):
Answer: "I don't have enough information in the provided context to answer this."
Output:
NONE

Now do the same for this actual answer:

Answer:
\"\"\"{answer}\"\"\"

Output (one claim per line, or NONE only if this is a refusal):
"""
    raw_text = llm_generate(None, prompt, max_new_tokens=500)
    print(f"    [DEBUG decompose_claims raw output]: {raw_text!r}")
    return _parse_claim_lines(raw_text)

def _parse_claim_lines(raw_text: str) -> list[str]:
    """Parse the line-based claim output: split on newlines, strip
    leading bullets/numbering the model might add despite being told
    not to, drop empty lines, and treat a bare NONE as an empty list."""
    stripped = raw_text.strip()
    if stripped.upper() == "NONE" or not stripped:
        return []

    lines = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() == "NONE":
            continue
        line = re.sub(r"^[\-\*\u2022]\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        if line:
            lines.append(line)
    return lines


def check_claims_support(claims: list[str], context_chunks: list[dict]) -> list[dict]:
    """Check each claim against the retrieved context, ONE CLAIM PER CALL
    (not batched). Slower (N calls instead of 1), but correctness
    matters more than call count here."""
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
    """Full pipeline: decompose answer into claims, check each against
    context, compute score = supported / total in code."""
    claims = decompose_claims(answer)
    if not claims:
        return {"score": None, "claims": [], "is_refusal": True}

    checked = check_claims_support(claims, context_chunks)
    supported_count = sum(1 for c in checked if c["supported"])
    score = supported_count / len(checked)
    return {"score": score, "claims": checked, "is_refusal": False}