"""Store codecs (pcm/store/codecs.py): round trips, error bounds, size accounting. CPU only."""
import math

import pytest
import torch

from pcm.store import codecs as C
from pcm.store.kv import KVCache, rebase_keys

L, H, D = 3, 2, 16
INV = 1.0 / (10000 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))


def cache(n=40, seed=0, scale=30.0):
    g = torch.Generator().manual_seed(seed)
    ks = [(torch.randn(1, H, n, D, generator=g) * scale).to(torch.bfloat16) for _ in range(L)]
    vs = [torch.randn(1, H, n, D, generator=g).to(torch.bfloat16) for _ in range(L)]
    return KVCache(ks, vs)


def test_bf16_is_exact_and_sized():
    c = cache()
    codec = C.make_codec("bf16")
    enc = codec.encode(c)
    out = codec.decode(enc)
    assert all(torch.equal(a, b) for a, b in zip(c.keys + c.values, out.keys + out.values))
    assert enc.nbytes == c.nbytes == 2 * L * H * 40 * D * 2


@pytest.mark.parametrize("bits", [8, 4, 2])
def test_pack_round_trip(bits):
    q = torch.randint(0, 1 << bits, (5, 7, 3), dtype=torch.uint8)
    p = C.pack_codes(q, bits)
    assert p.numel() == math.ceil(q.numel() * bits / 8)
    assert torch.equal(C.unpack_codes(p, bits, q.numel()).view_as(q), q)


@pytest.mark.parametrize("mode", ["token", "channel"])
@pytest.mark.parametrize("bits", [8, 4, 2])
def test_quantize_error_bound(mode, bits):
    x = torch.randn(H, 300, D) * 50
    q, mn, st = C.quantize(x, bits, mode, 128)
    y = C.dequantize(q, mn, st, mode, 128)
    idx = torch.arange(300) // 128
    step = st.float() if mode == "token" else st.float()[:, idx]
    # |error| <= step / 2, plus the float16 rounding of the minimum (clamped codes)
    assert bool(((y - x).abs() <= step * 0.5 + 0.06).all())
    assert q.shape == x.shape and int(q.max()) <= (1 << bits) - 1


@pytest.mark.parametrize("spec,expect", [("int8", (8, "channel", "token")), ("int4-ktvc", (4, "token", "channel")),
                                         ("int2-kcvc", (2, "channel", "channel")), ("int4-ktvt", (4, "token", "token"))])
def test_quant_codec_sizes(spec, expect):
    c = cache(n=256)
    codec = C.make_codec(spec, group=128)
    assert (codec.bits, codec.k_mode, codec.v_mode) == expect
    enc = codec.encode(c)
    bits, km, vm = expect
    elems = L * H * 256 * D
    groups = {"token": L * H * 256, "channel": L * H * 2 * D}
    assert enc.nbytes == 2 * elems * bits // 8 + (groups[km] + groups[vm]) * 2 * 2
    out = codec.decode(enc)
    err = C.relative_error(c, out)
    assert err["k"] < {8: 0.01, 4: 0.15, 2: 0.6}[bits] and err["v"] < {8: 0.01, 4: 0.15, 2: 0.6}[bits]
    assert out.keys[0].dtype == torch.bfloat16 and out.keys[0].shape == c.keys[0].shape


def test_rotation_to_zero_and_back():
    k = torch.randn(H, 50, D)
    pos = torch.arange(100, 150)
    back = C.rotate_positions(C.rotate_positions(k, pos, INV, -1), pos, INV, +1)
    assert torch.allclose(back, k, atol=1e-4)
    # rotating by +p equals the engine's re-basing by p (one position for all tokens)
    same = torch.full((50,), 7)
    assert torch.allclose(C.rotate_positions(k, same, INV, +1), rebase_keys(k, 7, INV), atol=1e-4)


def test_lowrank_full_rank_is_near_exact_and_rank_matters():
    pages = [(cache(n=64, seed=s), 10 + s) for s in range(6)]
    basis = C.LowRankBasis.fit(pages, INV, device="cpu")
    test = cache(n=64, seed=99)
    full = C.make_codec(f"lowrank{H * D}-int8", basis, INV)
    low = C.make_codec("lowrank4-int8", basis, INV)
    e_full = C.relative_error(test, full.decode(full.encode(test, start=5)))
    e_low = C.relative_error(test, low.decode(low.encode(test, start=5)))
    assert e_full["k"] < 0.02 and e_full["v"] < 0.02
    assert e_low["k"] > 5 * e_full["k"]
    assert full.shared_nbytes == 2 * L * (H * D * H * D + H * D) * 2
    enc = low.encode(test, start=5)
    assert enc.nbytes == 2 * (L * 64 * 4 * 8 // 8 + L * 1 * 4 * 2 * 2)


@pytest.mark.parametrize("spec", ["bf16+zstd-3", "bf16-shuffle+lz4", "int4+deflate", "int2-kcvc+lzma", "int8+ppmd", "int4+zstd-19"])
def test_lossless_equals_inner(spec):
    c = cache(n=130)
    codec = C.make_codec(spec)
    enc = codec.encode(c)
    out = codec.decode(enc, device="cpu")
    ref = codec.inner.decode(codec.inner.encode(c))
    assert all(torch.equal(a, b) for a, b in zip(ref.keys + ref.values, out.keys + out.values))
    assert enc.nbytes == len(enc.arrays["blob"])


def test_shuffle_round_trip():
    b = bytes(range(256)) * 4
    assert C.unshuffle_bytes(C.shuffle_bytes(b)) == b
    assert C.shuffle_bytes(b"\x01\x02\x03\x04") == b"\x01\x03\x02\x04"


def test_text_codec_ids_round_trip():
    t = C.TextCodec(coder_name="zstd-3")
    enc = t.encode_ids([5, 151643, 0, 42], start=3)
    assert t.ids(enc).tolist() == [5, 151643, 0, 42]
    assert C.TextCodec().encode_ids([1, 2, 3]).nbytes == 12
