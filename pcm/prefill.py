"""Chunked prefill, greedy generation and blockwise next-token loss.

A long sequence is never fed in one forward call: blocks of at most `block`
tokens are run against the cache of everything before them. The LM head is
applied only where logits are needed.
"""
from __future__ import annotations

import time
from typing import Sequence

import torch

from pcm.store.kv import KVCache, library_cache_length, new_library_cache


def _as_row(ids, device) -> torch.Tensor:
    """Token ids as a [1, n] LongTensor on `device`."""
    t = torch.as_tensor(ids, dtype=torch.long)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2 or t.shape[0] != 1:
        raise ValueError("expected one sequence of token ids")
    return t.to(device)


TRIM_SLACK_BYTES = 1 << 30


def _trim_allocator(device) -> None:
    """Return freed blocks to the driver when the allocator holds much more than is in use.

    Each prefill block needs slightly larger temporaries than the one before (the
    attention's expanded keys, values and mask grow with the cache), so the
    allocator cannot reuse the blocks it just freed and its reserved memory climbs
    far above what is allocated. Trimming keeps reserved close to allocated.
    """
    if device.type != "cuda":
        return
    if torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) > TRIM_SLACK_BYTES:
        torch.cuda.empty_cache()


def _library_cache(model, past: KVCache | None, new_tokens: int):
    """A preallocated library cache sized for `past` plus `new_tokens`."""
    if past is None:
        return new_library_cache(model.config, capacity=new_tokens)
    return past.to_library(model.config, capacity=past.n_tokens + new_tokens)


def _forward_blocks(model, ids: torch.Tensor, lib, block: int, position_offset: int, want_logits: bool):
    """Run ids through the model block by block on top of `lib`. Returns last-position logits or None."""
    n = ids.shape[1]
    if n == 0:
        raise ValueError("no tokens to prefill")
    if block < 1:
        raise ValueError("block must be positive")
    start_pos = library_cache_length(lib) + position_offset
    logits = None
    for s in range(0, n, block):
        e = min(s + block, n)
        positions = torch.arange(start_pos + s, start_pos + e, device=ids.device).unsqueeze(0)
        last = e == n
        if want_logits and last:
            out = model(
                input_ids=ids[:, s:e],
                position_ids=positions,
                past_key_values=lib,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[0, -1]
        else:
            # The decoder alone: no LM head for positions whose logits are not needed.
            model.model(input_ids=ids[:, s:e], position_ids=positions, past_key_values=lib, use_cache=True)
        if e - s > 1:
            _trim_allocator(ids.device)
    return logits


@torch.no_grad()
def prefill_in_blocks(
    model,
    input_ids,
    block: int = 2048,
    past: KVCache | None = None,
    position_offset: int = 0,
    return_logits: bool = False,
):
    """Encode `input_ids` after `past` in blocks of at most `block` tokens.

    Token i is given position len(past) + position_offset + i. `past` is not
    modified: it is copied once into a buffer sized for past + input, and each
    block is written into that buffer. Returns the cache of past + input; with
    return_logits=True, also the logits at the last position (a 1D tensor over
    the vocabulary).
    """
    ids = _as_row(input_ids, model.device)
    lib = _library_cache(model, past, ids.shape[1])
    logits = _forward_blocks(model, ids, lib, block, position_offset, return_logits)
    cache = KVCache.from_library(lib)
    return (cache, logits) if return_logits else cache


@torch.no_grad()
def generate_greedy(
    model,
    cache: KVCache | None,
    prompt_ids,
    max_new_tokens: int,
    stop_ids: Sequence[int] = (),
    block: int = 2048,
    timings: dict | None = None,
) -> list[int]:
    """Greedy continuation of `prompt_ids` given `cache`. Returns the new token ids.

    Generation stops before a token in `stop_ids` (which is not returned) or
    after max_new_tokens. The caller's cache is not modified. With a `timings`
    dict, the device is synchronised at three points and the dict receives
    prefill_s (start to first generated token chosen: time to first token),
    decode_s (the remaining steps) and new_tokens.
    """
    if timings is not None:
        _sync(model.device)
        t0 = time.perf_counter()
    ids = _as_row(prompt_ids, model.device)
    lib = _library_cache(model, cache, ids.shape[1] + max_new_tokens)
    logits = _forward_blocks(model, ids, lib, block, 0, True)
    stops = set(int(s) for s in stop_ids)
    out: list[int] = []
    for step_i in range(max_new_tokens):
        token = int(logits.argmax())  # .item() waits for the device
        if timings is not None and step_i == 0:
            t1 = time.perf_counter()
        if token in stops:
            break
        out.append(token)
        if len(out) == max_new_tokens:
            break
        step = torch.tensor([[token]], dtype=torch.long, device=ids.device)
        logits = _forward_blocks(model, step, lib, 1, 0, True)
    if timings is not None:
        _sync(model.device)
        t2 = time.perf_counter()
        t1 = t1 if max_new_tokens > 0 else t2
        timings.update({"prefill_s": t1 - t0, "decode_s": t2 - t1, "new_tokens": len(out)})
    return out


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def nll_in_blocks(model, input_ids, block: int = 2048) -> torch.Tensor:
    """Next-token negative log-likelihood of every token but the first.

    Returns a float32 CPU tensor of length n - 1: entry i is the loss of token
    i + 1 given tokens 0..i. The LM head runs on one block at a time.
    """
    ids = _as_row(input_ids, model.device)
    n = ids.shape[1]
    lib = new_library_cache(model.config, capacity=n)
    losses = []
    for s in range(0, n, block):
        e = min(s + block, n)
        positions = torch.arange(s, e, device=ids.device).unsqueeze(0)
        hidden = model.model(
            input_ids=ids[:, s:e], position_ids=positions, past_key_values=lib, use_cache=True
        ).last_hidden_state[0]
        targets = ids[0, s + 1 : e + 1]  # the next token of each position; one short in the last block
        logits = model.lm_head(hidden[: targets.shape[0]]).float()
        losses.append(torch.nn.functional.cross_entropy(logits, targets, reduction="none").cpu())
        del hidden, logits
        _trim_allocator(ids.device)
    return torch.cat(losses)


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, min_p: float, generator: torch.Generator) -> int:
    """One token drawn from `logits` with temperature, top-k, top-p and min-p, applied in that order
    (the order of the library's logits processors)."""
    x = logits.float() / temperature
    if top_k and top_k > 0:
        kth = torch.topk(x, min(int(top_k), x.shape[-1])).values[-1]
        x = x.masked_fill(x < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_x, order = torch.sort(x, descending=False)
        cum = torch.softmax(sorted_x, dim=-1).cumsum(dim=-1)
        drop = cum <= (1 - top_p)
        drop[-1] = False  # keep at least the most probable token
        x = x.masked_fill(drop.scatter(0, order, drop), float("-inf"))
    probs = torch.softmax(x, dim=-1)
    if min_p and min_p > 0:
        probs = probs.masked_fill(probs < min_p * probs.max(), 0.0)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate_sampled(
    model,
    cache: KVCache | None,
    prompt_ids,
    max_new_tokens: int,
    stop_ids: Sequence[int] = (),
    block: int = 2048,
    temperature: float = 0.7,
    top_k: int = 20,
    top_p: float = 0.8,
    min_p: float = 0.0,
    seed: int = 0,
    timings: dict | None = None,
) -> list[int]:
    """Like generate_greedy, but each token is sampled with a generator seeded by `seed` (reproducible)."""
    if timings is not None:
        _sync(model.device)
        t0 = time.perf_counter()
    gen = torch.Generator(device=model.device).manual_seed(int(seed) % (2**63))
    ids = _as_row(prompt_ids, model.device)
    lib = _library_cache(model, cache, ids.shape[1] + max_new_tokens)
    logits = _forward_blocks(model, ids, lib, block, 0, True)
    stops = set(int(s) for s in stop_ids)
    out: list[int] = []
    t1 = None
    for step_i in range(max_new_tokens):
        token = sample_token(logits, temperature, top_k, top_p, min_p, gen)
        if timings is not None and step_i == 0:
            t1 = time.perf_counter()
        if token in stops:
            break
        out.append(token)
        if len(out) == max_new_tokens:
            break
        step = torch.tensor([[token]], dtype=torch.long, device=ids.device)
        logits = _forward_blocks(model, step, lib, 1, 0, True)
    if timings is not None:
        _sync(model.device)
        t2 = time.perf_counter()
        t1 = t1 if t1 is not None else t2
        timings.update({"prefill_s": t1 - t0, "decode_s": t2 - t1, "new_tokens": len(out)})
    return out
