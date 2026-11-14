"""T4: YaRN through load_reader. The model accepts 100K positions; next-token loss is recorded.

Every condition ends on the same token of the same book, so the last `tail_tokens`
tokens are scored in all of them and only the amount of context before them differs.
"""
import math

import pytest
import torch

from conftest import free_cuda

pytestmark = pytest.mark.gpu

GIB = 1024**3


def run_condition(model, window, n, tail, block):
    from pcm.prefill import nll_in_blocks

    free_cuda()
    torch.cuda.reset_peak_memory_stats()
    ids = window[-n:]
    nll = nll_in_blocks(model, ids, block=block)
    assert nll.shape[0] == n - 1
    return {
        "tokens": n,
        "max_position": n - 1,
        "loss_all": float(nll.mean()),
        "loss_tail": float(nll[-tail:].mean()),
        "finite": bool(torch.isfinite(nll).all()),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
    }


def test_t4_yarn_accepts_long_positions(reader_repo, cuda, doc_ids, cfg, record):
    from pcm.models import load_reader, rope_summary

    t4, block = cfg["t4"], cfg["block"]
    factor, tail = t4["yarn_factor"], t4["tail_tokens"]
    window = doc_ids[: t4["long_tokens"]]
    assert len(window) == t4["long_tokens"]
    out = {"yarn_factor": factor, "tail_tokens": tail, "block": block, "conditions": {}}

    model, _ = load_reader(reader_repo, dtype=torch.bfloat16, device=cuda)
    out["rope_native"] = rope_summary(model)
    original = out["rope_native"]["max_position_embeddings"]
    out["conditions"]["native_16k"] = run_condition(model, window, t4["native_tokens"], tail, block)
    out["conditions"]["no_yarn_48k"] = run_condition(model, window, t4["mid_tokens"], tail, block)
    del model
    free_cuda()

    model, _ = load_reader(reader_repo, yarn_factor=factor, dtype=torch.bfloat16, device=cuda)
    out["rope_yarn"] = rope_summary(model)
    assert out["rope_yarn"]["rope_type"] == "yarn"
    assert out["rope_yarn"]["factor"] == factor
    assert out["rope_yarn"]["original_max_position_embeddings"] == original
    assert out["rope_yarn"]["rope_theta"] == out["rope_native"]["rope_theta"]
    assert factor * original >= t4["long_tokens"]
    out["conditions"]["yarn_48k"] = run_condition(model, window, t4["mid_tokens"], tail, block)
    out["conditions"]["yarn_100k"] = run_condition(model, window, t4["long_tokens"], tail, block)
    del model
    free_cuda()

    out["passed"] = all(c["finite"] for c in out["conditions"].values())
    record("T4", out)
    for name, c in out["conditions"].items():
        assert c["finite"], name
        assert not math.isnan(c["loss_tail"]), name
