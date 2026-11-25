"""KV mode (A6): bias mask, page store, key scaling, stitched layout, and the model-level checks.

CPU tests check the pieces as arithmetic. GPU tests load the development reader in float32 and check:
the explicit 4D additive mask reproduces the library's own causal masking (the open question of A1);
a page encoded right after the prefix equals the sequential encoding; a bias of log S on context
columns equals scaling those columns' attention weights by S; dividing keys by T equals dividing
those columns' scores by T; and KV mode through the runner keeps text mode's pages and token count.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pcm.expand import expand_kv, expand_text, kv_layout
from pcm.prefill import bias_mask
from pcm.store.kv import KVCache
from pcm.store.pages import PageStore


def _cache(n_layers=3, heads=2, n=5, d=4, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return KVCache([torch.randn(1, heads, n, d, generator=g).to(dtype) for _ in range(n_layers)],
                   [torch.randn(1, heads, n, d, generator=g).to(dtype) for _ in range(n_layers)])


# ---------------------------------------------------------------- CPU
def test_bias_mask_is_causal_and_biases_the_columns():
    m = bias_mask(3, 2, (1, 3, -0.5), torch.float32, "cpu")[0, 0]
    low = torch.finfo(torch.float32).min
    assert m.shape == (2, 5)
    assert m[0].tolist() == [0.0, -0.5, -0.5, 0.0, low]     # query at position 3 sees 0..3
    assert m[1].tolist() == [0.0, -0.5, -0.5, 0.0, 0.0]     # query at position 4 sees 0..4
    assert bias_mask(0, 1, (0, 0, 1.0), torch.bfloat16, "cpu").dtype == torch.bfloat16


@pytest.mark.parametrize("gpu_bytes", [10**9, 0])
def test_page_store_round_trip_is_bit_exact(tmp_path, gpu_bytes):
    store = PageStore(gpu_bytes, disk_dir=tmp_path / "pages", device="cpu")
    pages = {i: _cache(seed=i, n=3 + i) for i in range(3)}
    where = [store.put(i, c) for i, c in pages.items()]
    assert where == (["gpu"] * 3 if gpu_bytes else ["disk"] * 3)
    for i, c in pages.items():
        back = store.get(i)
        assert all(torch.equal(a, b) for a, b in zip(back.keys, c.keys)) and all(torch.equal(a, b) for a, b in zip(back.values, c.values))
        assert back.dtype == torch.bfloat16 and back.n_tokens == c.n_tokens
    assert (store.bytes_disk > 0) == (gpu_bytes == 0) and len(store) == 3
    with pytest.raises(KeyError):
        store.put(0, pages[0])
    store.clear()
    assert len(store) == 0 and not any((tmp_path / "pages").glob("*.npy"))


def test_page_store_spills_to_disk_when_the_gpu_budget_is_full(tmp_path):
    c = _cache()
    store = PageStore(c.nbytes, disk_dir=tmp_path, device="cpu")
    assert [store.put(i, _cache(seed=i)) for i in range(3)] == ["gpu", "disk", "disk"]


def test_scale_keys_touches_only_the_range():
    c = _cache(dtype=torch.float32)
    s = c.scale_keys(1, 3, 2.0)
    for k0, k1, v0, v1 in zip(c.keys, s.keys, c.values, s.values):
        assert torch.equal(k1[:, :, 1:3], k0[:, :, 1:3] * 2) and torch.equal(k1[:, :, [0, 3, 4]], k0[:, :, [0, 3, 4]])
        assert v1 is v0
    assert c.scale_keys(0, 5, 1.0).keys[0] is c.keys[0]
    with pytest.raises(ValueError):
        c.scale_keys(2, 9, 2.0)


def test_kv_layout_matches_text_mode_tokens():
    ctx = torch.arange(100, 140)
    gap = torch.tensor([7, 7])
    pages = [(10, 15), (15, 20), (30, 35)]
    layout = kv_layout(pages)
    assert layout == [("page", 10, 15), ("page", 15, 20), ("gap", 0, 0), ("page", 30, 35)]
    seq = torch.cat([ctx[a:b] if kind == "page" else gap for kind, a, b in layout])
    text, gaps = expand_text(ctx, [[10, 20], [30, 35]], gap)
    assert torch.equal(seq, text) and gaps == 1
    with pytest.raises(ValueError):
        kv_layout([(0, 5), (3, 8)])


def test_expand_kv_stitches_rebases_and_scales():
    torch.manual_seed(0)
    d = 4
    inv = 1.0 / (10000 ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    p, a, b = _cache(n=2, d=d, dtype=torch.float32), _cache(n=3, d=d, dtype=torch.float32, seed=1), _cache(n=2, d=d, dtype=torch.float32, seed=2)
    cache, n_ctx = expand_kv(p, [a, b], inv, temperature=0.5)
    assert n_ctx == 5 and cache.n_tokens == 7
    plain, _ = expand_kv(p, [a, b], inv)
    assert torch.allclose(cache.keys[0][:, :, 2:], plain.keys[0][:, :, 2:] * 2.0) and torch.equal(cache.keys[0][:, :, :2], p.keys[0])
    # Page b moved from position 2 (right after the prefix) to position 5: its keys are rotated by 3 positions.
    from pcm.store.kv import rebase_keys

    assert torch.allclose(plain.keys[1][:, :, 5:], rebase_keys(b.keys[1], 3, inv))


# ---------------------------------------------------------------- GPU, development reader in float32
def _need_gpu(request):
    from pcm.health import EXIT_GPU_LOST, gpu_status

    if request.config.getoption("--no-gpu"):
        pytest.skip("GPU tests skipped on request (--no-gpu)")
    st = gpu_status()
    if not st["ok"]:
        pytest.exit(f"no usable GPU: {st['reason']}", returncode=EXIT_GPU_LOST)


@pytest.fixture(scope="module")
def dev_reader(request):
    _need_gpu(request)
    from pcm.models import load_reader

    model, tok = load_reader("reader_dev", dtype=torch.float32)
    yield model, tok
    del model
    torch.cuda.empty_cache()


def _ids(tok, text):
    return torch.tensor(tok(text, add_special_tokens=False)["input_ids"])


TEXT_P = "You answer questions about a story.\nStory:\n"
TEXT_A = "Mary went to the kitchen. John picked up the football there. The weather was cold all day."
TEXT_B = "Then John went to the garden and dropped the football next to the old oak tree."
TEXT_Q = "\nQuestion: Where is the football?\nAnswer:"


@pytest.mark.gpu
def test_explicit_4d_mask_reproduces_the_library_causal_mask(dev_reader):
    from pcm.prefill import _forward_blocks, _library_cache, prefill_in_blocks

    model, tok = dev_reader
    ctx = prefill_in_blocks(model, _ids(tok, TEXT_P + TEXT_A))
    q = _ids(tok, TEXT_Q).to(model.device)[None]
    plain = _forward_blocks(model, q, _library_cache(model, ctx, q.shape[1]), 2048, 0, True)
    masked = _forward_blocks(model, q, _library_cache(model, ctx, q.shape[1]), 2048, 0, True, bias=(0, ctx.n_tokens, 0.0))
    assert torch.allclose(plain, masked, atol=1e-4, rtol=1e-4)
    assert int(plain.argmax()) == int(masked.argmax())


@pytest.mark.gpu
def test_one_page_after_the_prefix_equals_sequential_encoding(dev_reader):
    from pcm.prefill import prefill_in_blocks
    from pcm.store.kv import encode_page, rope_inv_freq

    model, tok = dev_reader
    p_ids, a_ids = _ids(tok, TEXT_P), _ids(tok, TEXT_A)
    P = prefill_in_blocks(model, p_ids)
    seq = prefill_in_blocks(model, torch.cat([p_ids, a_ids]))
    kv, n_ctx = expand_kv(P, [encode_page(model, P, a_ids)], rope_inv_freq(model))
    assert n_ctx == len(a_ids)
    for k0, k1, v0, v1 in zip(seq.keys, kv.keys, seq.values, kv.values):
        assert torch.allclose(k0, k1, atol=2e-3 * k0.abs().max().item()) and torch.allclose(v0, v1, atol=2e-3 * v0.abs().max().item())


def _attention_to(model, cache, q_ids, cols, bias=None):
    """Attention weights of the last query token on cached columns `cols`, every layer and head (eager)."""
    from pcm.prefill import _library_cache, bias_mask

    model.set_attn_implementation("eager")
    try:
        lib = _library_cache(model, cache, q_ids.shape[1])
        n0 = cache.n_tokens
        pos = torch.arange(n0, n0 + q_ids.shape[1], device=model.device)[None]
        mask = bias_mask(n0, q_ids.shape[1], bias if bias is not None else (0, 0, 0.0), model.dtype, model.device)
        out = model.model(input_ids=q_ids, position_ids=pos, past_key_values=lib, use_cache=True, output_attentions=True, attention_mask=mask)
        return torch.stack([a[0, :, -1, cols[0]:cols[1]] for a in out.attentions]), torch.stack([a[0, :, -1, :] for a in out.attentions])
    finally:
        model.set_attn_implementation("sdpa")


@pytest.mark.gpu
def test_log_s_bias_scales_context_attention_weights_by_s(dev_reader):
    """With all hidden states equal, adding log S to context scores multiplies their unnormalised weights by S.
    Checked on layer 0, whose queries depend only on the question tokens."""
    from pcm.prefill import prefill_in_blocks

    model, tok = dev_reader
    p_ids, a_ids = _ids(tok, TEXT_P), _ids(tok, TEXT_A)
    cache = prefill_in_blocks(model, torch.cat([p_ids, a_ids]))
    q = _ids(tok, TEXT_Q).to(model.device)[None]
    cols = (len(p_ids), len(p_ids) + len(a_ids))
    s = 0.8
    _, w1 = _attention_to(model, cache, q, cols)
    _, w2 = _attention_to(model, cache, q, cols, bias=(cols[0], cols[1], math.log(s)))
    l0a, l0b = w1[0].double(), w2[0].double()               # layer 0: [heads, kv]
    ratio = (l0b / l0a)
    inside = ratio[:, cols[0]:cols[1]]
    outside = torch.cat([ratio[:, :cols[0]], ratio[:, cols[1]:]], dim=1)
    # Each row: inside / outside = S (up to the shared normaliser).
    rel = inside.mean(dim=1) / outside.mean(dim=1)
    assert torch.allclose(rel, torch.full_like(rel, s), rtol=1e-3)


@pytest.mark.gpu
def test_key_temperature_divides_context_scores(dev_reader):
    """Keys of context tokens divided by T: the layer-0 log-weights of those columns, relative to a prefix column,
    are the plain ones divided by T."""
    from pcm.prefill import prefill_in_blocks

    model, tok = dev_reader
    p_ids, a_ids = _ids(tok, TEXT_P), _ids(tok, TEXT_A)
    cache = prefill_in_blocks(model, torch.cat([p_ids, a_ids]))
    q = _ids(tok, TEXT_Q).to(model.device)[None]
    cols = (len(p_ids), len(p_ids) + len(a_ids))
    t = 0.8
    _, w1 = _attention_to(model, cache, q, cols)
    _, w2 = _attention_to(model, cache.scale_keys(cols[0], cols[1], 1 / t), q, cols)
    s1 = torch.log(w1[0].double())
    s2 = torch.log(w2[0].double())
    ref = 0                                                  # first prefix column: score unchanged in both
    d1 = s1[:, cols[0]:cols[1]] - s1[:, ref:ref + 1]
    d2 = s2[:, cols[0]:cols[1]] - s2[:, ref:ref + 1]
    # score_ctx / T - score_ref, against score_ctx - score_ref: recover score_ctx - score_ref from both.
    # d2 = (d1 + score_ref) / T - score_ref, so d2 - d1 / T = score_ref (1 / T - 1) is the same for every column.
    resid = d2 - d1 / t
    assert torch.allclose(resid, resid[:, :1].expand_as(resid), atol=1e-3)


# ---------------------------------------------------------------- run script: cells and the tuning rule
def _run_a6(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("run_a6", root / "scripts" / "run_a6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a5 = yaml.safe_load((root / "configs" / "a5_pillars.yaml").read_text(encoding="utf-8"))
    text = {**a5["pcm"], "page": 512, "budget": 16384, "name": "pcm-text"}
    (tmp_path / "pcm_text.yaml").write_text(yaml.safe_dump({"pcm_text": text}), encoding="utf-8")
    cfg = yaml.safe_load((root / "configs" / "a6_kv.yaml").read_text(encoding="utf-8"))
    cfg["pcm_text"] = str(tmp_path / "pcm_text.yaml")
    (tmp_path / "a6.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return mod, mod.load_config(tmp_path / "a6.yaml"), text


def test_run_a6_cells(tmp_path):
    from pcm.tasks.common import config_hash

    mod, cfg, text = _run_a6(tmp_path)
    assert config_hash(mod.mode_config(cfg, "text")) == config_hash(text)        # same configuration as A5: answers reusable
    kv = mod.mode_config(cfg, "kv")
    assert kv["method"] == "pcm_kv" and kv["kv"]["temperature"] == 1.0 and kv["page"] == 512 and kv["name"] == "pcm-kv"
    tune = mod.tune_cells(cfg)
    assert len(tune) == 4 * 3 * 3 and all(c["split"] == "train" and c["seed"] == 1000 for c in tune)
    without = mod.e6_cells(cfg, None)
    assert {c["method"] for c in without} == {"pcm-text", "pcm-kv", "pcm-text-p128", "pcm-text-p2048", "pcm-text-p4096"}
    cells = mod.e6_cells(cfg, {"temperature": 0.8, "scale": 0.9})
    main = 3 * (13 * 4 + 4)
    pages = 4 * 2 * 4 * 2 - 4 * 2 * 2   # page 512 is the main setting: its text and kv-ape cells are already counted
    assert len(cells) == main + pages
    ape = next(c for c in cells if c["method"] == "pcm-kv-ape-p2048")["mcfg"]
    assert ape["page"] == 2048 and ape["kv"] == {"temperature": 0.8, "scale": 0.9, "store_gpu_gib": 8.0}


def test_run_a6_tuning_rule(tmp_path):
    from pcm.tasks.common import config_hash

    mod, cfg, _ = _run_a6(tmp_path)
    tr = cfg["ape"]["train"]
    rows = []
    for c in mod.tune_cells(cfg):
        acc = {(0.8, 0.9): 0.7, (0.9, 1.0): 0.7, (1.0, 1.0): 0.6}.get((c["T"], c["S"]), 0.5)
        h = config_hash(c["mcfg"])
        rows += [{"method_hash": h, "task": c["task"], "status": "ok", "score": float(i < acc * tr["n"])} for i in range(tr["n"])]
    ch = mod.choose_ape(cfg, rows)
    assert (ch["temperature"], ch["scale"]) == (0.9, 1.0)                    # tie at 0.7: nearer to (1, 1)
    with pytest.raises(RuntimeError, match="incomplete"):
        mod.choose_ape(cfg, rows[:-1])
