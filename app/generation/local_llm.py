"""
Local Generation Backend: Qwen2.5-3B-Instruct.

Inference configuration:
- Architecture & Memory: Runs at bfloat16 (~6GB weights, unquantized) within 8GB VRAM
  alongside BGE-M3 and the CrossEncoder reranker. Running unquantized directly avoids
  external quantization kernel dependencies on Blackwell (sm_120) architectures.
- Determinism: `do_sample=False` (greedy decoding) ensures reproducible, deterministic
  evaluations across runs.
- Token Prefill: Supports explicit prompt token prefilling (e.g., forcing JSON start `[`)
  for structured generation tasks.
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