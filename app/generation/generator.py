"""
Generation.

Takes retrieved chunks + a user query, generates an answer via the
Claude API, constrained to answer ONLY from retrieved context, with an
explicit "not enough information" fallback -- this fallback behavior is
itself what the no_answer QA category tests (refusal vs hallucination).
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client: Anthropic | None = None
_MODEL = "claude-sonnet-4-5-20250929"

_SYSTEM_PROMPT = """You are a documentation assistant answering questions about FastAPI \
using ONLY the provided context chunks. 

Rules:
- Answer only using information present in the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, respond with \
exactly: "I don't have enough information in the provided context to answer this." \
Do not guess or fill gaps with general knowledge.
- Be concise and direct. Cite which part of the context supports your answer where natural.
"""


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def generate_answer(query: str, retrieved_chunks: list[dict]) -> dict:
    """Generate a grounded answer from retrieved chunks.

    Returns {"answer": str, "context_used": list[dict]} -- context_used
    is passed through so faithfulness scoring downstream doesn't need to
    re-fetch it.
    """
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source_path']} > {c['header_path']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    user_message = f"Context:\n\n{context_text}\n\n---\n\nQuestion: {query}"

    client = _get_client()
    response = client.messages.create(
        model=_MODEL,
        max_tokens=500,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer_text = response.content[0].text

    return {"answer": answer_text, "context_used": retrieved_chunks}