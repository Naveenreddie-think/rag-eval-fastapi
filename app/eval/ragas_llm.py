"""
Step 5: Minimal Claude wrapper for RAGAS's LLM interface.

RAGAS (0.1.21, the version confirmed to install cleanly for this
project) defaults to OpenAI via langchain wrappers. Rather than install
langchain_anthropic (which pulled in a newer langchain_core and broke
ragas's own pinned <0.3 requirement during testing), this implements
RAGAS's minimal BaseRagasLLM interface directly against our own
Anthropic client -- avoids the dependency conflict entirely by not
routing through langchain's Anthropic integration at all.
"""

import os

from anthropic import Anthropic, AsyncAnthropic
from dotenv import load_dotenv
from langchain_core.outputs import Generation, LLMResult
from ragas.llms.base import BaseRagasLLM

load_dotenv()

_MODEL = "claude-sonnet-4-5-20250929"


class ClaudeRagasLLM(BaseRagasLLM):
    def __init__(self):
        super().__init__()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=api_key)
        self._async_client = AsyncAnthropic(api_key=api_key)

    def generate_text(self, prompt, n: int = 1, temperature: float = 1e-8,
                       stop=None, callbacks=None) -> LLMResult:
        text = prompt.to_string()
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": text}],
        )
        return LLMResult(generations=[[Generation(text=response.content[0].text)]])

    async def agenerate_text(self, prompt, n: int = 1, temperature=None,
                              stop=None, callbacks=None) -> LLMResult:
        text = prompt.to_string()
        response = await self._async_client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": text}],
        )
        return LLMResult(generations=[[Generation(text=response.content[0].text)]])


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