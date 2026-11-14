"""CPU unit tests of the KV adapter and of key re-basing. No model is loaded."""
import pytest
import torch

from pcm.store.kv import KVCache, rebase_keys


def default_inv_freq(dim: int = 128, theta: float = 1_000_000.0) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))


def random_cache(layers=3, heads=2, tokens=10, dim=8, seed=0) -> KVCache:
    g = torch.Generator().manual_seed(seed)
    keys = [torch.randn(1, heads, tokens, dim, generator=g) for _ in range(layers)]
    values = [torch.randn(1, heads, tokens, dim, generator=g) for _ in range(layers)]
    return KVCache(keys, values)


def reference_rotate(k: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """The library's formula, written out: k * cos + rotate_half(k) * sin."""
    freqs = positions.double()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    half = k.shape[-1] // 2
    rot = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return k.double() * emb.cos() + rot.double() * emb.sin()


def test_shapes_and_sizes():
    c = random_cache(layers=3, heads=2, tokens=10, dim=8)
    assert c.n_layers == 3
    assert c.n_tokens == 10 == len(c)
    assert c.nbytes == 2 * 3 * 2 * 10 * 8 * 4
    k, v = c.layer(1)
    assert k.shape == v.shape == (1, 2, 10, 8)


def test_slice_then_concat_is_identity():
    c = random_cache()
    a, b, d = c.slice(0, 4), c.slice(4, 7), c.slice(7, 10)
    assert (a.n_tokens, b.n_tokens, d.n_tokens) == (4, 3, 3)
    back = KVCache.concat([a, b, d])
    for i in range(c.n_layers):
        assert torch.equal(back.keys[i], c.keys[i])
        assert torch.equal(back.values[i], c.values[i])


def test_slice_rejects_bad_ranges():
    c = random_cache(tokens=10)
    with pytest.raises(ValueError):
        c.slice(5, 3)
    with pytest.raises(ValueError):
        c.slice(0, 11)


def test_concat_rejects_mismatched_layers():
    with pytest.raises(ValueError):
        KVCache.concat([random_cache(layers=2), random_cache(layers=3)])


def test_clone_is_independent():
    c = random_cache()
    d = c.slice(2, 5).clone()
    d.keys[0].zero_()
    assert c.keys[0][:, :, 2:5].abs().sum() > 0


def test_to_dtype_and_back():
    c = random_cache()
    h = c.to(dtype=torch.float16)
    assert h.dtype == torch.float16 and c.dtype == torch.float32
    assert h.nbytes * 2 == c.nbytes


def test_library_round_trip_shares_memory():
    c = random_cache()
    lib = c.to_library()
    assert lib.get_seq_length() == c.n_tokens
    back = KVCache.from_library(lib)
    assert back.n_layers == c.n_layers and back.n_tokens == c.n_tokens
    for i in range(c.n_layers):
        assert back.keys[i].data_ptr() == c.keys[i].data_ptr()
        assert back.values[i].data_ptr() == c.values[i].data_ptr()


def test_library_update_does_not_modify_source():
    c = random_cache()
    before = [k.clone() for k in c.keys]
    lib = c.to_library()
    lib.update(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8), 0)
    assert lib.layers[0].keys.shape[-2] == c.n_tokens + 1
    assert torch.equal(c.keys[0], before[0]) and c.n_tokens == 10


def test_rebase_zero_is_identity():
    k = torch.randn(1, 2, 5, 128, dtype=torch.float64)
    assert torch.allclose(rebase_keys(k, 0, default_inv_freq()), k, atol=1e-12)


def test_rebase_matches_reference_rotation():
    inv_freq = default_inv_freq()
    raw = torch.randn(1, 2, 6, 128, dtype=torch.float64)
    pos = torch.arange(6)
    at_0 = reference_rotate(raw, pos, inv_freq)
    at_5000 = reference_rotate(raw, pos + 5000, inv_freq)
    assert torch.allclose(rebase_keys(at_0, 5000, inv_freq), at_5000, atol=1e-9)


def test_rebase_composes_and_inverts():
    inv_freq = default_inv_freq()
    k = torch.randn(1, 2, 4, 128, dtype=torch.float64)
    twice = rebase_keys(rebase_keys(k, 1234, inv_freq), 4321, inv_freq)
    assert torch.allclose(twice, rebase_keys(k, 5555, inv_freq), atol=1e-9)
    assert torch.allclose(rebase_keys(rebase_keys(k, 777, inv_freq), -777, inv_freq), k, atol=1e-9)


def test_rebase_preserves_norm_and_dtype():
    inv_freq = default_inv_freq()
    k = torch.randn(1, 2, 4, 128)
    out = rebase_keys(k, 31000, inv_freq)
    assert out.dtype == k.dtype and out.shape == k.shape
    assert torch.allclose(out.norm(dim=-1), k.norm(dim=-1), rtol=1e-5)
    half = rebase_keys(k.to(torch.bfloat16), 31000, inv_freq)
    assert half.dtype == torch.bfloat16


def test_cache_rebase_touches_keys_only():
    c = random_cache(dim=128)
    r = c.rebase(100, default_inv_freq())
    for i in range(c.n_layers):
        assert not torch.allclose(r.keys[i], c.keys[i])
        assert r.values[i] is c.values[i]


def test_buffered_library_cache_writes_in_place_and_keeps_source():
    c = random_cache(layers=2, heads=2, tokens=10, dim=8)
    before = c.keys[0].clone()
    lib = c.to_library(capacity=13)
    buffer_ptr = lib.layers[0].keys.data_ptr()
    assert lib.get_seq_length() == 10
    assert buffer_ptr != c.keys[0].data_ptr()  # copied once, not shared
    new_k, new_v = torch.ones(1, 2, 3, 8), torch.full((1, 2, 3, 8), 2.0)
    k, v = lib.update(new_k, new_v, 0)
    assert k.shape[-2] == 13 and k.data_ptr() == buffer_ptr  # same buffer, no reallocation
    assert torch.equal(k[:, :, :10], before) and torch.equal(k[:, :, 10:], new_k)
    assert torch.equal(v[:, :, 10:], new_v)
    assert torch.equal(c.keys[0], before) and c.n_tokens == 10
    back = KVCache.from_library(lib)
    assert back.keys[0].shape[-2] == 13 and back.keys[1].shape[-2] == 10


def test_buffered_library_cache_grows_when_capacity_is_exceeded():
    c = random_cache(layers=1, heads=2, tokens=4, dim=8)
    lib = c.to_library(capacity=4)
    k, _ = lib.update(torch.ones(1, 2, 5, 8), torch.ones(1, 2, 5, 8), 0)
    assert k.shape[-2] == 9 and lib.get_seq_length() == 9
    assert torch.equal(k[:, :, :4], c.keys[0])


def test_buffered_capacity_must_cover_the_cache():
    with pytest.raises(ValueError):
        random_cache(tokens=10).to_library(capacity=5)
