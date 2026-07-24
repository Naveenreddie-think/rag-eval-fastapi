"""
Step 5: Custom faithfulness/groundedness metric.

Built alongside RAGAS (not instead of it) so the metric is explainable
rather than a black box: decompose the generated answer into atomic
factual claims, then check each claim against the retrieved context.
faithfulness_score = (# claims supported by context) / (# total claims),
computed in code from real per-claim judgments -- never asking the model
to self-report the aggregate score directly.

This mirrors what RAGAS's faithfulness metric does internally (claim
extraction + per-claim NLI-style check against context), so running
both side by side lets us compare where they agree/disagree, and
explain exactly what our own number means if asked.
"""

import json
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Anthropic | None = None
_MODEL = "claude-sonnet-4-5-20250929"

REFUSAL_PHRASE = "I don't have enough information in the provided context to answer this."


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def is_refusal_response(answer: str) -> bool:
    """Direct check for whether the model actually refused, independent
    of claim decomposition. Needed because a refusal WITH honest
    additional explanation (e.g. "I don't have enough info... [context]
    only covers X, not Y") still produces extractable claims from the
    explanation, so decompose_claims()'s empty-list signal alone
    undercounts genuine refusals. Checks for the exact phrase the system
    prompt requires (see app/generation/generator.py's _SYSTEM_PROMPT)."""
    return REFUSAL_PHRASE in answer


def _extract_json_array(text, raw_response_for_debug):
    r"""Parse a JSON array out of a model response defensively.

    Real bug this fixes: an earlier naive re.search(r"\[.*\]", text,
    re.DOTALL) matched from the FIRST '[' to the LAST ']' in the text.
    When the model's reasoning included code containing brackets (e.g.
    "Annotated[str, Depends(get_cookie_or_token)]"), that earlier
    unrelated bracket became the greedy match's start, swallowing all
    the reasoning prose in between into one non-JSON blob and failing
    to parse -- even though the model DID include a valid array, just
    with other bracket-containing text before it.

    Fix: find all NON-NESTED bracket spans (no brackets inside), then
    try parsing them in REVERSE order (last match in the text first),
    since the instructed answer format puts the real array at the end,
    after any reasoning/preamble the model added despite being told not
    to. Returns the first span that parses successfully.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    candidates = re.findall(r"\[[^\[\]]*\]", stripped)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"Could not parse a JSON array out of model response. "
        f"Raw response was:\n{raw_response_for_debug!r}"
    )


def decompose_claims(answer: str) -> list[str]:
    """Break an answer into a list of atomic, independently-checkable
    factual claims. Returns a plain list of claim strings. Uses
    assistant-message prefill ("[") to force the model to start directly
    with the array, rather than relying on instruction-following alone."""
    prompt = f"""Break the following answer into a list of atomic, independently \
checkable factual claims. Each claim should be a single, simple statement. \
If the answer is a refusal (e.g. states it doesn't have enough information), \
return an empty list.

Answer:
\"\"\"{answer}\"\"\"

Respond with ONLY a JSON array of strings, no other text. Example:
["FastAPI uses Pydantic for data validation.", "TestClient is built on HTTPX."]
"""
    client = _get_client()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "["},
        ],
    )
    # prefill means the response continues from "[" -- prepend it back
    raw_text = "[" + response.content[0].text
    return _extract_json_array(raw_text, raw_text)


def check_claims_support(claims: list[str], context_chunks: list[dict]) -> list[dict]:
    """Check each claim against the retrieved context in a single batched
    call. Returns [{"claim": str, "supported": bool}, ...]. Uses the same
    prefill technique as decompose_claims()."""
    if not claims:
        return []

    context_text = "\n\n---\n\n".join(c["text"] for c in context_chunks)
    claims_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))

    prompt = f"""Context:
\"\"\"{context_text}\"\"\"

For each numbered claim below, determine if it is directly supported by \
the context above (true) or not supported/not present in the context (false). \
Be strict: the claim must actually follow from the context, not just be \
plausible or generally true.

Claims:
{claims_list}

Respond with ONLY a JSON array of booleans, in the same order, no other text. \
Example: [true, false, true]
"""
    client = _get_client()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "["},
        ],
    )
    raw_text = "[" + response.content[0].text
    supported_flags = _extract_json_array(raw_text, raw_text)

    assert len(supported_flags) == len(claims), (
        f"Claim/support count mismatch: {len(claims)} claims, "
        f"{len(supported_flags)} support flags. Raw response: {raw_text!r}"
    )
    return [{"claim": c, "supported": s} for c, s in zip(claims, supported_flags)]


def custom_faithfulness_score(answer: str, context_chunks: list[dict]) -> dict:
    """Full pipeline: decompose answer into claims, check each against
    context, compute score = supported / total in code.

    Returns {"score": float, "claims": [...], "is_refusal": bool}.
    is_refusal=True (score=None) when the answer has zero claims, which
    should only happen for genuine refusals -- worth checking this
    matches the no_answer category, not silently treating a malformed
    answer as a "perfect" empty-claim-list case.
    """
    claims = decompose_claims(answer)
    if not claims:
        return {"score": None, "claims": [], "is_refusal": True}

    checked = check_claims_support(claims, context_chunks)
    supported_count = sum(1 for c in checked if c["supported"])
    score = supported_count / len(checked)
    return {"score": score, "claims": checked, "is_refusal": False}