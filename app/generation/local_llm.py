"""
Local generation model: Qwen2.5-3B-Instruct, replacing Claude API calls
across generator.py, faithfulness.py, and ragas_llm.py.

Reason for this specific choice: given available VRAM (8GB, shared with
BGE-M3 + the cross-encoder reranker) and the GPU's bleeding-edge
architecture (RTX 5060, Blackwell/sm_120 -- already required PyTorch
nightly cu128 to work at all for embeddings), a 3B model at bf16
(~6GB weights, no quantization) avoids stacking a NEW unverified
compatibility risk (bitsandbytes on Blackwell) on top of an already
fragile setup. Same transformers/torch stack already confirmed working,
so this is a drop-in extension, not a new toolchain.

do_sample=False (greedy decoding) is used throughout for reproducible,
deterministic outputs -- appropriate for eval, where we want the same
input to reliably produce the same output across re-runs.

Supports an optional `prefill` string (forces the response to start with
given text, e.g. "[") -- mirrors the assistant-message-prefill trick
that fixed Claude's occasional "shows its work instead of returning
JSON" behavior in faithfulness.py. With a local model we have full
control over the prompt tokens, so this is done by directly appending
the prefill text to the prompt before generation, rather than needing
an API-specific prefill mechanism.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading {_MODEL_NAME} (first call only, may take a while)...")
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        print("Model loaded.")
    return _model, _tokenizer


def generate(system_prompt: str | None, user_message: str,
             max_new_tokens: int = 500, prefill: str | None = None) -> str:
    """Generate a response from Qwen2.5-3B-Instruct.

    system_prompt: optional system instruction (None to omit).
    user_message: the user turn content.
    prefill: optional text to force the response to start with (e.g. "[").
    Returns the generated text (with prefill prepended back if used).
    """
    model, tokenizer = _load()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prefill:
        prompt_text += prefill

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return (prefill + text) if prefill else text