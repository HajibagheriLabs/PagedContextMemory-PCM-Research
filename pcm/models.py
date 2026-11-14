"""Reader loading (with optional YaRN) and chat prompt construction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import torch

from pcm.hub import model_path

CONTEXT_SENTINEL = "\x00PCM_CONTEXT\x00"


# ---- RoPE / YaRN configuration ----
def rope_api(config) -> str:
    """Which RoPE configuration style the installed transformers uses."""
    return "rope_parameters" if hasattr(config, "rope_parameters") else "rope_scaling"


def shipped_rope(model_dir) -> dict:
    """rope_theta and the original maximum positions, read from the shipped config.json."""
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    params = raw.get("rope_parameters") or {}
    theta = params.get("rope_theta", raw.get("rope_theta"))
    if theta is None:
        raise ValueError("shipped config has no rope_theta")
    return {"rope_theta": float(theta), "max_position_embeddings": int(raw["max_position_embeddings"])}


def apply_yarn(config, factor: float, rope_theta: float, original_max_positions: int) -> None:
    """Configure YaRN on `config` in place, through whichever API the library has."""
    if rope_api(config) == "rope_parameters":
        config.rope_parameters = {
            "rope_type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(original_max_positions),
            "rope_theta": float(rope_theta),
        }
    else:
        config.rope_theta = float(rope_theta)
        config.rope_scaling = {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(original_max_positions),
        }
    config.max_position_embeddings = int(original_max_positions * factor)


def yarn_factor_for(max_tokens: int, original_max_positions: int, choices=(2, 4, 8, 16)) -> int | None:
    """Smallest factor whose extended range covers max_tokens; None if no extension is needed."""
    if max_tokens <= original_max_positions:
        return None
    for f in choices:
        if f * original_max_positions >= max_tokens:
            return f
    raise ValueError(f"{max_tokens} tokens exceed {choices[-1]} x {original_max_positions}")


def rope_summary(model) -> dict:
    """RoPE settings of a loaded model, as the model itself holds them."""
    config = model.config
    rot = model.model.rotary_emb
    if rope_api(config) == "rope_parameters":
        params = dict(config.rope_parameters)
    else:
        params = dict(getattr(config, "rope_scaling", None) or {})
        params.setdefault("rope_theta", getattr(config, "rope_theta", None))
    return {
        "api": rope_api(config),
        "rope_type": params.get("rope_type", params.get("type", "default")),
        "rope_theta": params.get("rope_theta"),
        "factor": params.get("factor"),
        "original_max_position_embeddings": params.get("original_max_position_embeddings"),
        "max_position_embeddings": config.max_position_embeddings,
        "attention_scaling": float(getattr(rot, "attention_scaling", 1.0)),
        "inv_freq_min": float(rot.inv_freq.min()),
        "inv_freq_max": float(rot.inv_freq.max()),
    }


def load_reader(
    model_id: str,
    yarn_factor: float | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device="cuda:0",
    attn_implementation: str = "sdpa",
):
    """Load a reader and its tokenizer from the local cache. Returns (model, tokenizer).

    model_id is a key of configs/assets.yaml or a repo id. With yarn_factor, RoPE
    is replaced by YaRN with that factor; rope_theta and the original length come
    from the shipped config. The model is frozen and in eval mode.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    path = model_path(model_id)
    config = AutoConfig.from_pretrained(path)
    if yarn_factor is not None:
        shipped = shipped_rope(path)
        apply_yarn(config, yarn_factor, shipped["rope_theta"], shipped["max_position_embeddings"])
    tokenizer = AutoTokenizer.from_pretrained(path)
    kwargs = {"config": config, "attn_implementation": attn_implementation}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:  # older transformers name for the same argument
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **kwargs)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


# ---- prompts ----
@dataclass(frozen=True)
class Chat:
    """A chat prompt split around the context.

    prefix: everything before the context (system turn, instruction, context opener).
    context: the context text.
    suffix: everything after it (question, end of the user turn, assistant header
            with thinking disabled).
    Token ids are always built per segment (prefix ids + context ids + suffix ids),
    so the same prefix and suffix ids serve text mode and KV mode.
    """

    prefix: str
    context: str
    suffix: str

    @property
    def text(self) -> str:
        return self.prefix + self.context + self.suffix


def build_chat(tokenizer, prefix_parts: Sequence[str], context_text: str, question: str) -> Chat:
    """Build the prompt with the tokenizer's chat template, thinking disabled.

    prefix_parts: the system prompt first (empty or missing: no system turn), then
    any number of strings placed in the user turn before the context (task
    instruction, context opener), joined by blank lines. The question follows the
    context after a blank line.
    """
    parts = [p for p in prefix_parts]
    system = parts[0] if parts else ""
    before = [p for p in parts[1:] if p]
    user = "\n\n".join(before) + ("\n" if before else "") + CONTEXT_SENTINEL + "\n\n" + question
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    if rendered.count(CONTEXT_SENTINEL) != 1:
        raise ValueError("chat template did not keep the context placeholder exactly once")
    prefix, suffix = rendered.split(CONTEXT_SENTINEL)
    return Chat(prefix=prefix, context=context_text, suffix=suffix)


def encode_text(tokenizer, text: str) -> torch.Tensor:
    """Token ids of `text` as a 1D LongTensor, with no special tokens added."""
    return torch.tensor(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=torch.long)


def stop_ids(tokenizer) -> list[int]:
    """End-of-turn and end-of-text token ids."""
    ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    return sorted(i for i in ids if isinstance(i, int) and i >= 0)
