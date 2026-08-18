"""
Step 5: Local Qwen wrapper for RAGAS's LLM interface.

RAGAS (0.1.21, the version confirmed to install cleanly for this
project) defaults to OpenAI via langchain wrappers. This implements
RAGAS's minimal BaseRagasLLM interface directly against our own local
model, avoiding the dependency conflict entirely by not routing through
langchain's model integrations at all.
"""

from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM


class LocalQwenRagasLLM(BaseRagasLLM):
    """Local Qwen2.5-3B-Instruct wrapper for RAGAS BaseRagasLLM interface."""
    def __init__(self):
        super().__init__()
        from app.generation.local_llm import generate as _gen
        self._generate = _gen

    def generate_text(self, prompt, n: int = 1, temperature: float = 1e-8,
                       stop=None, callbacks=None) -> LLMResult:
        text = prompt.to_string()
        out = self._generate(None, text, max_new_tokens=1024)
        return LLMResult(generations=[[Generation(text=out)]])

    async def agenerate_text(self, prompt, n: int = 1, temperature=None,
                              stop=None, callbacks=None) -> LLMResult:
        text = prompt.to_string()
        out = self._generate(None, text, max_new_tokens=1024)
        return LLMResult(generations=[[Generation(text=out)]])