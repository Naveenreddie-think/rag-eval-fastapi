"""
Local LLM Adapter for RAGAS Evaluation Framework.

Implements RAGAS's BaseRagasLLM interface using the local Qwen2.5-3B-Instruct model,
enabling offline, reproducible RAGAS evaluation without third-party API dependencies.
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