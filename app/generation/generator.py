"""
Generation.

Responsibility: take the top-k retrieved chunks + user query, generate an
answer via the Claude API, constrained to answer only from retrieved
context, with an explicit "not enough information" fallback path.

To be implemented:
- generate_answer(query, retrieved_chunks) -> {answer, citations}
- The "no answer in corpus" fallback is itself an eval category (Step 4/5)
  -- worth a dedicated system prompt instruction, not an afterthought.
"""
