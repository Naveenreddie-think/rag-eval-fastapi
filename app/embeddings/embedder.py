"""
Step 2: Embedding.

Responsibility: wrap whichever embedding model we choose, provide a single
embed(texts: list[str]) -> list[vector] interface so the rest of the
pipeline doesn't care which model is behind it (needed for Step 5's
embedding-model sweep).

Choice not locked yet -- to be decided with a stated reason once Step 2
starts (candidates: OpenAI text-embedding-3-small, a local sentence-
transformers model, or Voyage AI).
"""
