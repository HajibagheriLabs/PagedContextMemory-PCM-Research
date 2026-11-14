"""Engine invariants in float32: T1 (chunked prefill), T2 (key re-basing), T3 (stitching)."""
import pytest
import torch

from conftest import free_cuda

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def model(reader_repo, cuda):
    from pcm.models import load_reader

    model, _ = load_reader(reader_repo, dtype=torch.float32, device=cuda)
    yield model
    del model
    free_cuda()


def per_layer_max_abs(a, b) -> list[float]:
    return [float((x.double() - y.double()).abs().max()) for x, y in zip(a, b)]


def test_t1_chunked_prefill_equals_one_shot_f32(model, doc_ids, cfg, record):
    from pcm.prefill import prefill_in_blocks

    n, block = cfg["t1"]["tokens"], cfg["block"]
    ids = doc_ids[:n]
    one, logits_one = prefill_in_blocks(model, ids, block=n, return_logits=True)
    blk, logits_blk = prefill_in_blocks(model, ids, block=block, return_logits=True)

    assert one.n_tokens == blk.n_tokens == n
    assert one.n_layers == blk.n_layers == model.config.num_hidden_layers
    logit_diff = float((logits_one.float() - logits_blk.float()).abs().max())
    key_diff = max(per_layer_max_abs(one.keys, blk.keys))
    value_diff = max(per_layer_max_abs(one.values, blk.values))
    same_top1 = bool(logits_one.argmax() == logits_blk.argmax())
    record(
        "T1_f32",
        {
            "tokens": n,
            "block": block,
            "blocks": -(-n // block),
            "last_logit_max_abs_diff": logit_diff,
            "threshold": cfg["t1"]["f32_max_abs_logit_diff"],
            "cache_key_max_abs_diff": key_diff,
            "cache_value_max_abs_diff": value_diff,
            "logit_abs_max": float(logits_one.abs().max()),
            "same_top1": same_top1,
            "passed": logit_diff < cfg["t1"]["f32_max_abs_logit_diff"],
        },
    )
    assert same_top1
    assert logit_diff < cfg["t1"]["f32_max_abs_logit_diff"]


def rebase_experiment(model, page, delta, block):
    """Encode `page` at positions 0.. and delta.., re-base the first; per-layer deviations."""
    from pcm.prefill import prefill_in_blocks
    from pcm.store.kv import rope_inv_freq

    at_0 = prefill_in_blocks(model, page, block=block, position_offset=0)
    at_delta = prefill_in_blocks(model, page, block=block, position_offset=delta)
    rebased = at_0.rebase(delta, rope_inv_freq(model))
    key_diff = per_layer_max_abs(rebased.keys, at_delta.keys)
    value_diff = per_layer_max_abs(at_0.values, at_delta.values)
    key_scale = [float(k.abs().max()) for k in at_delta.keys]
    value_scale = [float(v.abs().max()) for v in at_delta.values]
    return {
        "at_0": at_0,
        "at_delta": at_delta,
        "rebased": rebased,
        "key_diff": key_diff,
        "value_diff": value_diff,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "key_rel": [d / s for d, s in zip(key_diff, key_scale)],
        "value_rel": [d / s for d, s in zip(value_diff, value_scale)],
        "unrebased_diff": per_layer_max_abs(at_0.keys, at_delta.keys),
    }


@pytest.fixture(scope="module")
def t2(model, doc_ids, cfg):
    return rebase_experiment(model, doc_ids[: cfg["t2"]["page_tokens"]], cfg["t2"]["delta"], cfg["block"])


def test_t2_rebasing_is_exact_without_prefix(t2, cfg, record):
    """T2 exactly as specified: float32, all layers, max abs diff < 1e-3."""
    tol = cfg["t2"]["max_abs_diff"]
    passed = max(t2["key_diff"]) < tol and max(t2["value_diff"]) < tol
    record(
        "T2",
        {
            "page_tokens": cfg["t2"]["page_tokens"],
            "delta": cfg["t2"]["delta"],
            "threshold": tol,
            "key_max_abs_diff_per_layer": t2["key_diff"],
            "value_max_abs_diff_per_layer": t2["value_diff"],
            "key_max_abs_diff": max(t2["key_diff"]),
            "value_max_abs_diff": max(t2["value_diff"]),
            "key_abs_max_per_layer": t2["key_scale"],
            "value_abs_max_per_layer": t2["value_scale"],
            "key_max_abs_diff_without_rebasing": max(t2["unrebased_diff"]),
            "passed": passed,
        },
    )
    # Without re-basing the keys must differ: otherwise the test would prove nothing.
    assert max(t2["unrebased_diff"]) > 100 * tol
    assert max(t2["value_diff"]) < tol
    assert max(t2["key_diff"]) < tol


def test_t2_relative_and_functional(model, t2, doc_ids, cfg, record):
    """Diagnostic for T2: deviation relative to the tensors' scale, and its effect on logits."""
    from pcm.prefill import prefill_in_blocks

    d = cfg["t2_diagnostic"]
    n, delta, block = cfg["t2"]["page_tokens"], cfg["t2"]["delta"], cfg["block"]
    # A continuation read through the re-based cache and through the directly encoded one.
    follow = doc_ids[n : n + d["continuation_tokens"]]
    _, via_rebased = prefill_in_blocks(
        model, follow, block=block, past=t2["rebased"], position_offset=delta, return_logits=True
    )
    _, via_direct = prefill_in_blocks(
        model, follow, block=block, past=t2["at_delta"], position_offset=delta, return_logits=True
    )
    # The same continuation with the cache left at its original positions: must be visibly wrong.
    _, via_unrebased = prefill_in_blocks(
        model, follow, block=block, past=t2["at_0"], position_offset=delta, return_logits=True
    )
    logit_diff = float((via_rebased - via_direct).abs().max())
    logit_diff_unrebased = float((via_unrebased - via_direct).abs().max())
    key_rel, value_rel = max(t2["key_rel"]), max(t2["value_rel"])
    passed = (
        key_rel < d["max_rel_diff"]
        and value_rel < d["max_rel_diff"]
        and logit_diff < d["max_abs_logit_diff"]
        and bool(via_rebased.argmax() == via_direct.argmax())
    )
    record(
        "T2_relative",
        {
            "key_rel_diff_per_layer": t2["key_rel"],
            "value_rel_diff_per_layer": t2["value_rel"],
            "key_max_rel_diff": key_rel,
            "value_max_rel_diff": value_rel,
            "threshold_rel": d["max_rel_diff"],
            "continuation_tokens": len(follow),
            "last_logit_max_abs_diff": logit_diff,
            "last_logit_max_abs_diff_without_rebasing": logit_diff_unrebased,
            "threshold_logit": d["max_abs_logit_diff"],
            "same_top1": bool(via_rebased.argmax() == via_direct.argmax()),
            "passed": passed,
        },
    )
    assert key_rel < d["max_rel_diff"]
    assert value_rel < d["max_rel_diff"]
    assert via_rebased.argmax() == via_direct.argmax()
    assert logit_diff < d["max_abs_logit_diff"]


def test_t3_stitching_sanity(model, tokenizer, doc_ids, cfg, record):
    from pcm.models import build_chat, encode_text
    from pcm.prefill import prefill_in_blocks
    from pcm.store.kv import encode_page, rope_inv_freq, stitch

    t3, block = cfg["t3"], cfg["block"]
    n = t3["page_tokens"]
    pages = [doc_ids[s : s + n] for s in t3["page_starts"]]
    chat = build_chat(tokenizer, [t3["system"], t3["instruction"], t3["opener"]], "", t3["question"])
    prefix_ids = encode_text(tokenizer, chat.prefix)
    suffix_ids = encode_text(tokenizer, chat.suffix)
    assert chat.text == chat.prefix + chat.context + chat.suffix
    assert "<think>" in chat.suffix  # thinking disabled: the template closes an empty think block

    # KV mode
    inv_freq = rope_inv_freq(model)
    prefix_cache = prefill_in_blocks(model, prefix_ids, block=block)
    page_caches = [encode_page(model, prefix_cache, p, block=block) for p in pages]
    for pc in page_caches:
        assert pc.n_tokens == n
    stitched = stitch(prefix_cache, page_caches, inv_freq)
    kv_final, kv_logits = prefill_in_blocks(model, suffix_ids, block=block, past=stitched, return_logits=True)

    # Equivalent text prompt
    text_ids = torch.cat([prefix_ids, *pages, suffix_ids])
    text_final, text_logits = prefill_in_blocks(model, text_ids, block=block, return_logits=True)

    p = len(prefix_ids)
    assert stitched.n_tokens == p + 2 * n
    assert kv_final.n_tokens == text_final.n_tokens == len(text_ids)
    assert torch.isfinite(kv_logits).all()

    # Position layout. Layer-0 keys and values depend only on the token and its position,
    # so they must agree with the text prompt at every position of the stitched cache.
    # Deviations are relative to the largest magnitude in the compared tensor.
    tol = t3["max_rel_diff"]
    k0_scale = float(text_final.keys[0].abs().max())
    v0_scale = float(text_final.values[0].abs().max())
    k0_diff = float((kv_final.keys[0] - text_final.keys[0]).abs().max())
    v0_diff = float((kv_final.values[0] - text_final.values[0]).abs().max())
    # The first page sits right after the prefix in both modes: identical at every layer.
    first = slice(p, p + n)
    first_page_rel = max(
        d / float(k[:, :, first].abs().max())
        for d, k in zip(
            per_layer_max_abs([k[:, :, first] for k in kv_final.keys], [k[:, :, first] for k in text_final.keys]),
            text_final.keys,
        )
    )
    # The second page never attended to the first one: a real deviation is expected above layer 0.
    second = slice(p + n, p + 2 * n)
    second_page_rel = [
        d / float(k[:, :, second].abs().max())
        for d, k in zip(
            per_layer_max_abs([k[:, :, second] for k in kv_final.keys], [k[:, :, second] for k in text_final.keys]),
            text_final.keys,
        )
    ]
    second_page_fro = [
        float((a[:, :, second].double() - b[:, :, second].double()).norm() / b[:, :, second].double().norm())
        for a, b in zip(kv_final.keys, text_final.keys)
    ]
    logit_diff = float((kv_logits.float() - text_logits.float()).abs().max())
    passed = k0_diff / k0_scale < tol and v0_diff / v0_scale < tol and first_page_rel < tol
    record(
        "T3",
        {
            "prefix_tokens": p,
            "page_tokens": n,
            "pages": len(pages),
            "page_starts": t3["page_starts"],
            "suffix_tokens": len(suffix_ids),
            "total_tokens_kv": kv_final.n_tokens,
            "total_tokens_text": text_final.n_tokens,
            "question_positions_kv": [stitched.n_tokens, kv_final.n_tokens - 1],
            "question_positions_text": [p + 2 * n, text_final.n_tokens - 1],
            "logits_finite": bool(torch.isfinite(kv_logits).all()),
            "layer0_key_max_rel_diff": k0_diff / k0_scale,
            "layer0_value_max_rel_diff": v0_diff / v0_scale,
            "first_page_key_max_rel_diff_all_layers": first_page_rel,
            "second_page_key_max_rel_diff_per_layer": second_page_rel,
            "second_page_key_max_rel_diff": max(second_page_rel),
            "second_page_key_frobenius_rel_diff_per_layer": second_page_fro,
            "last_logit_max_abs_diff_kv_vs_text": logit_diff,
            "same_top1_kv_vs_text": bool(kv_logits.argmax() == text_logits.argmax()),
            "top1_kv": tokenizer.decode([int(kv_logits.argmax())]),
            "top1_text": tokenizer.decode([int(text_logits.argmax())]),
            "threshold_rel": tol,
            "passed": passed,
        },
    )
    assert k0_diff / k0_scale < tol
    assert v0_diff / v0_scale < tol
    assert first_page_rel < tol


def test_t2_exact_in_float64(model, doc_ids, cfg, record):
    """Diagnostic for T2: with float64 arithmetic and float64 rotary angles the re-based
    keys equal the directly encoded ones to rounding error, at every layer.

    The library computes rotary angles in float32 whatever the model dtype. For this
    test only, the rotary module is patched to compute them in float64, so that the
    comparison isolates the algebra of re-basing from that rounding.
    """
    from pcm.store.kv import rotary_module

    tol = cfg["t2_diagnostic"]["f64_max_rel_diff"]
    rot = rotary_module(model)
    original_forward = rot.forward

    def forward_f64(x, position_ids):
        freqs = position_ids[..., None].double() * rot.inv_freq.to(device=x.device, dtype=torch.float64)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * rot.attention_scaling).to(x.dtype), (emb.sin() * rot.attention_scaling).to(x.dtype)

    page = doc_ids[: cfg["t2"]["page_tokens"]]
    try:
        model.double()
        f64_library_angles = rebase_experiment(model, page, cfg["t2"]["delta"], cfg["block"])
        rot.forward = forward_f64
        f64 = rebase_experiment(model, page, cfg["t2"]["delta"], cfg["block"])
    finally:
        rot.forward = original_forward
        model.float()
    passed = max(f64["key_rel"]) < tol and max(f64["value_rel"]) < tol
    record(
        "T2_float64",
        {
            "threshold_rel": tol,
            "key_max_abs_diff": max(f64["key_diff"]),
            "value_max_abs_diff": max(f64["value_diff"]),
            "key_max_rel_diff": max(f64["key_rel"]),
            "value_max_rel_diff": max(f64["value_rel"]),
            "key_max_abs_diff_per_layer": f64["key_diff"],
            "value_max_abs_diff_per_layer": f64["value_diff"],
            "float64_model_float32_angles_key_max_abs_diff": max(f64_library_angles["key_diff"]),
            "float64_model_float32_angles_value_max_abs_diff": max(f64_library_angles["value_diff"]),
            "float64_model_float32_angles_key_max_rel_diff": max(f64_library_angles["key_rel"]),
            "key_max_abs_diff_without_rebasing": max(f64["unrebased_diff"]),
            "passed": passed,
        },
    )
    assert max(f64["key_rel"]) < tol
    assert max(f64["value_rel"]) < tol
