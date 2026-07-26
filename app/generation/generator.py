"""
Generation.

Takes retrieved chunks + a user query, generates an answer via the
local Qwen2.5-3B-Instruct model (app/generation/local_llm.py),
constrained to answer ONLY from retrieved context, with an explicit
"not enough information" fallback -- this fallback behavior is itself
what the no_answer QA category tests (refusal vs hallucination).

Switched from Claude API to a local model when Anthropic API credits
ran out and the decision was made not to renew -- verified via
scripts/verify_local_llm.py that Qwen2.5-3B-Instruct correctly exhibits
the refusal behavior before wiring it in here.
"""

from app.generation.local_llm import generate as llm_generate

_SYSTEM_PROMPT = """You are a documentation assistant answering questions about FastAPI \
using ONLY the provided context chunks.

Rules:
- Answer only using information present in the context below. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, respond with \
exactly: "I don't have enough information in the provided context to answer this." \
Do not guess or fill gaps with general knowledge.
- Be concise and direct. Cite which part of the context supports your answer where natural.
"""


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

    answer_text = llm_generate(_SYSTEM_PROMPT, user_message, max_new_tokens=500)

    return {"answer": answer_text, "context_used": retrieved_chunks}