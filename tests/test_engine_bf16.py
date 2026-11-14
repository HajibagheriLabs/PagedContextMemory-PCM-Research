"""Engine invariants in bf16: T1 (top-1 agreement), greedy generation, T5 (memory guard)."""
import time

import pytest
import torch

from conftest import free_cuda

pytestmark = pytest.mark.gpu

GIB = 1024**3


@pytest.fixture(scope="module")
def model(reader_repo, cuda):
    from pcm.models import load_reader

    model, _ = load_reader(reader_repo, dtype=torch.bfloat16, device=cuda)
    yield model
    del model
    free_cuda()


@pytest.mark.records("T1_bf16")
def test_t1_chunked_prefill_equals_one_shot_bf16(model, tokenizer, doc_ids, cfg, record):
    from pcm.prefill import prefill_in_blocks

    n, block = cfg["t1"]["tokens"], cfg["block"]
    ids = doc_ids[:n]
    _, logits_one = prefill_in_blocks(model, ids, block=n, return_logits=True)
    _, logits_blk = prefill_in_blocks(model, ids, block=block, return_logits=True)
    top_one, top_blk = int(logits_one.argmax()), int(logits_blk.argmax())
    record(
        "T1_bf16",
        {
            "tokens": n,
            "block": block,
            "top1_one_shot": tokenizer.decode([top_one]),
            "top1_chunked": tokenizer.decode([top_blk]),
            "last_logit_max_abs_diff": float((logits_one.float() - logits_blk.float()).abs().max()),
            "passed": top_one == top_blk,
        },
    )
    assert top_one == top_blk


@pytest.mark.records("generate_greedy")
def test_generate_greedy_matches_library(model, tokenizer, record):
    from pcm.models import build_chat, encode_text, stop_ids
    from pcm.prefill import generate_greedy, prefill_in_blocks

    chat = build_chat(tokenizer, [], "", "What is the capital of France?")
    ids = encode_text(tokenizer, chat.text)
    # Split the prompt: the first part is prefilled into a cache, the rest is the prompt.
    cut = len(ids) // 2
    cache = prefill_in_blocks(model, ids[:cut])
    ours = generate_greedy(model, cache, ids[cut:], max_new_tokens=16, stop_ids=stop_ids(tokenizer))

    with torch.no_grad():
        out = model.generate(
            ids.unsqueeze(0).to(model.device),
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    theirs = [t for t in out[0, len(ids):].tolist() if t not in stop_ids(tokenizer)]
    text = tokenizer.decode(ours, skip_special_tokens=True)
    record("generate_greedy", {"answer": text, "matches_library_generate": ours == theirs, "passed": ours == theirs})
    assert ours == theirs
    assert "Paris" in text
    assert cache.n_tokens == cut  # the caller's cache is not modified


@pytest.mark.records("T5")
def test_t5_memory_guard(model, doc_ids, cfg, record):
    from pcm.prefill import prefill_in_blocks

    n, block, limit = cfg["t5"]["tokens"], cfg["block"], cfg["t5"]["max_peak_vram_gib"]
    ids = doc_ids[:n]
    free_cuda()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = prefill_in_blocks(model, ids, block=block)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    record(
        "T5",
        {
            "tokens": n,
            "block": block,
            "allocated_before_gib": weights / GIB,
            "cache_gib": cache.nbytes / GIB,
            "peak_vram_gib": peak / GIB,
            "peak_reserved_gib": reserved / GIB,
            "limit_gib": limit,
            "seconds": seconds,
            "tokens_per_second": n / seconds,
            "passed": peak / GIB < limit,
        },
    )
    assert cache.n_tokens == n
    assert peak / GIB < limit
