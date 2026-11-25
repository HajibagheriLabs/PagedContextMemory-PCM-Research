"""How far a stitched KV context is from sequential encoding of the same tokens (A6; D-048).

Both caches hold the same tokens at the same positions: P followed by the selected pages (and gap
markers) in document order. The sequential cache encodes them as one sequence, so every token
attended to everything before it; the stitched cache encoded each page after P only.

- KV deviation, per layer: ||K_stitched - K_sequential||_F / ||K_sequential||_F over the context tokens,
  and the same for V (CacheBlend's KV deviation, relative so that layers of different scale compare).
- Attention deviation, per layer: the question's attention, computed by the model on top of each
  cache (forward attention: from layer 1 on the question's hidden states differ too). For every head
  and question token, the total-variation distance between the two attention distributions over all
  columns (half the L1 distance; 0 = identical, 1 = disjoint), averaged over heads and question tokens.
- At the first answer position: KL(sequential || stitched) of the next-token distribution, and whether
  the top-1 tokens agree.
"""
from __future__ import annotations

import torch

from pcm.store.kv import KVCache


def kv_deviation(sequential: KVCache, stitched: KVCache, start: int, stop: int) -> list[dict]:
    """Per layer: relative Frobenius deviation of keys and values over tokens [start, stop)."""
    if sequential.n_tokens != stitched.n_tokens:
        raise ValueError("the caches hold different numbers of tokens")
    out = []
    for (k0, v0), (k1, v1) in zip(zip(sequential.keys, sequential.values), zip(stitched.keys, stitched.values)):
        a, b = k0[:, :, start:stop].float(), k1[:, :, start:stop].float()
        c, d = v0[:, :, start:stop].float(), v1[:, :, start:stop].float()
        out.append({"k": float((b - a).norm() / a.norm()), "v": float((d - c).norm() / c.norm())})
    return out


@torch.no_grad()
def question_pass(model, cache: KVCache, q_ids: torch.Tensor, bias=None) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run the question tokens on top of `cache` with eager attention. Returns (per-layer attention
    [heads, q, kv] in float32 on the GPU, next-token logits at the last question token)."""
    from pcm.prefill import _library_cache, bias_mask

    q = torch.as_tensor(q_ids, dtype=torch.long, device=model.device).reshape(1, -1)
    lib = _library_cache(model, cache, q.shape[1])
    n0 = cache.n_tokens
    pos = torch.arange(n0, n0 + q.shape[1], device=model.device)[None]
    mask = bias_mask(n0, q.shape[1], bias if bias is not None else (0, 0, 0.0), model.dtype, model.device)
    previous = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        out = model(input_ids=q, position_ids=pos, past_key_values=lib, use_cache=True, output_attentions=True,
                    attention_mask=mask, logits_to_keep=1)
    finally:
        model.set_attn_implementation(previous)
    return [a[0].float() for a in out.attentions], out.logits[0, -1].float()


def attention_deviation(att_a: list[torch.Tensor], att_b: list[torch.Tensor]) -> list[float]:
    """Per layer: total-variation distance between the two attention distributions, mean over heads and tokens."""
    return [float(0.5 * (a - b).abs().sum(dim=-1).mean()) for a, b in zip(att_a, att_b)]


def next_token_agreement(logits_ref: torch.Tensor, logits: torch.Tensor) -> dict:
    lp, lq = torch.log_softmax(logits_ref.double(), -1), torch.log_softmax(logits.double(), -1)
    return {"kl": float((lp.exp() * (lp - lq)).sum()), "top1_same": bool(int(logits_ref.argmax()) == int(logits.argmax()))}
