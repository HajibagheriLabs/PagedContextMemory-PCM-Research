"""Store codecs for page caches (specification, section 4.6; decisions D-049, D-050).

One interface for every codec:

    enc = codec.encode(cache, start)   # cache: KVCache of one page; start: position of its first token
    cache = codec.decode(enc)          # a KVCache of the same shape and dtype
    enc.nbytes                         # bytes stored for this page
    codec.shared_nbytes                # bytes stored once per store (low-rank bases), not per page

Codecs:

- `bf16`: the page as it is (reference). 114,688 bytes per token for Qwen3-1.7B.
- `int{8,4,2}-k{c,t}v{c,t}`: uniform asymmetric quantization (min and step per group, both stored as
  float16). `c` = per channel: one group per (head, channel) and run of `group` tokens; `t` = per token:
  one group per (head, token) over the head dimension. KIVI is keys per channel, values per token
  (`kcvt`); the swapped (`ktvc`) and same-granularity (`kcvc`, `ktvt`) settings are the others. Codes are
  counted packed (8 / bits per byte).
- `lowrank{r}-int{B}`: per layer, a PCA basis of the 1,024-dimensional per-token key vector (8 heads x 128)
  and value vector, fitted on calibration pages (`LowRankBasis.fit`). Keys are first rotated back to
  position 0 token by token (they are stored after RoPE), so one basis serves every position; decoding
  rotates them forward again. The r coefficients per token are quantized per component over runs of
  `group` tokens with B bits. The bases (float16) are the codec's shared bytes.
- `<codec>+<coder>`: the codec's stored arrays (packed codes, float16 parameters; raw bf16 bytes for `bf16`)
  serialized and compressed on the CPU with a lossless coder (zstd-3, zstd-19, lz4, lzma, deflate, ppmd).
  `bf16-shuffle+<coder>` first splits the bf16 bytes into high- and low-byte planes. Lossless, so the
  decoded cache equals the inner codec's.
- `text`: the page's token ids only (see `TextCodec`); its KV is recomputed on expansion.
"""
from __future__ import annotations

import io
import lzma
import math
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch

from pcm.store.kv import KVCache

FP16 = 2  # bytes of a stored float16 parameter


# ---------------------------------------------------------------- lossless coders (CPU)
def _zstd(level: int):
    import zstandard

    return (lambda b: zstandard.ZstdCompressor(level=level).compress(b),
            lambda b: zstandard.ZstdDecompressor().decompress(b))


def _lz4():
    import lz4.frame

    return lz4.frame.compress, lz4.frame.decompress


def _deflate():
    def comp(b):
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        return c.compress(b) + c.flush()

    def decomp(b):
        d = zlib.decompressobj(-15)
        return d.decompress(b) + d.flush()

    return comp, decomp


def _ppmd():
    import pyppmd

    return (lambda b: pyppmd.compress(b, max_order=6, mem_size=16 << 20),
            lambda b: pyppmd.decompress(b, max_order=6, mem_size=16 << 20))


CODERS = {
    "zstd-3": lambda: _zstd(3),
    "zstd-19": lambda: _zstd(19),
    "lz4": _lz4,
    "lzma": lambda: (lambda b: lzma.compress(b, preset=6), lzma.decompress),
    "deflate": _deflate,
    "ppmd": _ppmd,
}


def coder(name: str):
    """(compress, decompress) of a lossless coder: bytes -> bytes."""
    if name not in CODERS:
        raise ValueError(f"unknown coder {name!r}; known: {sorted(CODERS)}")
    return CODERS[name]()


def shuffle_bytes(b: bytes, itemsize: int = 2) -> bytes:
    """Byte planes: all first bytes of each item, then all second bytes, ... (as blosc's shuffle)."""
    a = np.frombuffer(b, dtype=np.uint8)
    if a.size % itemsize:
        raise ValueError("length is not a multiple of the item size")
    return a.reshape(-1, itemsize).T.copy().tobytes()


def unshuffle_bytes(b: bytes, itemsize: int = 2) -> bytes:
    a = np.frombuffer(b, dtype=np.uint8)
    return a.reshape(itemsize, -1).T.copy().tobytes()


# ---------------------------------------------------------------- bit packing (GPU or CPU)
def pack_codes(q: torch.Tensor, bits: int) -> torch.Tensor:
    """uint8 codes in [0, 2^bits) -> packed uint8, 8 / bits codes per byte (flattened, zero-padded)."""
    flat = q.reshape(-1).to(torch.uint8)
    if bits == 8:
        return flat.clone()
    per = 8 // bits
    pad = (-flat.numel()) % per
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.view(-1, per)
    out = torch.zeros(flat.shape[0], dtype=torch.uint8, device=q.device)
    for i in range(per):
        out |= flat[:, i] << (bits * i)
    return out


def unpack_codes(p: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    if bits == 8:
        return p[:numel]
    per = 8 // bits
    mask = (1 << bits) - 1
    parts = [(p >> (bits * i)) & mask for i in range(per)]
    return torch.stack(parts, dim=1).reshape(-1)[:numel]


# ---------------------------------------------------------------- uniform quantization
def quantize(x: torch.Tensor, bits: int, mode: str, group: int):
    """x: [H, n, D] (any float dtype). Returns (codes uint8 [H, n, D], minimum fp16, step fp16).

    mode "token": one group per (head, token) over D -> parameters [H, n, 1].
    mode "channel": one group per (head, channel) and run of `group` tokens -> parameters [H, ceil(n/group), D].
    The stored float16 minimum and step are the ones used to compute the codes, so decoding is exact
    up to the code's rounding (|error| <= step / 2, plus clamping at the float16-rounded minimum)."""
    levels = (1 << bits) - 1
    x = x.float()
    H, n, D = x.shape
    if mode == "token":
        xg = x.unsqueeze(2)  # [H, n, 1, D]: groups along D
        mn, mx = xg.amin(dim=-1, keepdim=True), xg.amax(dim=-1, keepdim=True)
    elif mode == "channel":
        nc = -(-n // group)
        pad = nc * group - n
        xp = torch.cat([x, x[:, -1:].expand(H, pad, D)], dim=1) if pad else x  # repeat the last token: min/max unchanged
        xg = xp.view(H, nc, group, D)
        mn, mx = xg.amin(dim=2, keepdim=True), xg.amax(dim=2, keepdim=True)
    else:
        raise ValueError(f"unknown quantization mode {mode!r}")
    mn16 = mn.half()
    step16 = ((mx - mn16.float()) / levels).clamp_min(1e-8).half()
    step16 = torch.where(step16.float() > 0, step16, torch.ones_like(step16))
    q = torch.clamp(torch.round((xg - mn16.float()) / step16.float()), 0, levels).to(torch.uint8)
    if mode == "token":
        return q.squeeze(2), mn16.squeeze(2), step16.squeeze(2)
    return q.reshape(H, -1, D)[:, :n], mn16.squeeze(2), step16.squeeze(2)


def dequantize(q: torch.Tensor, mn: torch.Tensor, step: torch.Tensor, mode: str, group: int) -> torch.Tensor:
    """Inverse of quantize, in float32: q [H, n, D]; parameters as quantize returns them."""
    H, n, D = q.shape
    if mode == "token":
        return q.float() * step.float() + mn.float()  # [H, n, 1] broadcast over D
    idx = torch.arange(n, device=q.device) // group
    return q.float() * step.float()[:, idx] + mn.float()[:, idx]


def n_groups(shape: tuple[int, int, int], mode: str, group: int) -> int:
    H, n, D = shape
    return H * n if mode == "token" else H * (-(-n // group)) * D


# ---------------------------------------------------------------- RoPE to and from position 0
def rotate_positions(K: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, sign: int) -> torch.Tensor:
    """K [..., n, D] float32: rotate token i by sign * positions[i] (the library's half-split layout)."""
    angle = positions.to(torch.float64)[:, None] * inv_freq.to(torch.float64).cpu()[None, :] * float(sign)
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1).to(device=K.device, dtype=torch.float32)
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1).to(device=K.device, dtype=torch.float32)
    half = K.shape[-1] // 2
    return K * cos + torch.cat((-K[..., half:], K[..., :half]), dim=-1) * sin


# ---------------------------------------------------------------- codecs
@dataclass
class Encoded:
    codec: str
    n_tokens: int
    start: int
    nbytes: int
    arrays: dict = field(default_factory=dict)   # name -> tensor (GPU) or bytes (CPU, lossless)
    meta: dict = field(default_factory=dict)


class Codec:
    name = "codec"
    shared_nbytes = 0
    lossless_inner = False

    def encode(self, cache: KVCache, start: int = 0) -> Encoded:
        raise NotImplementedError

    def decode(self, enc: Encoded) -> KVCache:
        raise NotImplementedError

    def to_bytes(self, enc: Encoded) -> bytes:
        """The stored arrays serialized (packed codes, float16 parameters), for lossless coding."""
        buf = io.BytesIO()
        for k in sorted(enc.arrays):
            a = enc.arrays[k]
            buf.write(a.contiguous().cpu().view(torch.uint8).numpy().tobytes() if isinstance(a, torch.Tensor) else a)
        return buf.getvalue()

    def from_bytes(self, data: bytes, like: Encoded, device) -> Encoded:
        arrays, off = {}, 0
        for k in sorted(like.arrays):
            a = like.arrays[k]
            n = a.numel() * a.element_size()
            t = torch.frombuffer(bytearray(data[off:off + n]), dtype=torch.uint8).view(a.dtype).reshape(a.shape)
            arrays[k] = t.to(device)
            off += n
        if off != len(data):
            raise ValueError("serialized length does not match the arrays")
        return Encoded(like.codec, like.n_tokens, like.start, like.nbytes, arrays, dict(like.meta))


class BF16(Codec):
    name = "bf16"

    def encode(self, cache, start=0):
        k = torch.stack([x[0] for x in cache.keys])   # [L, H, n, D]
        v = torch.stack([x[0] for x in cache.values])
        return Encoded(self.name, cache.n_tokens, start, k.numel() * k.element_size() + v.numel() * v.element_size(), {"k": k, "v": v})

    def decode(self, enc):
        k, v = enc.arrays["k"], enc.arrays["v"]
        return KVCache([k[i].unsqueeze(0) for i in range(k.shape[0])], [v[i].unsqueeze(0) for i in range(v.shape[0])])


class Quant(Codec):
    """Uniform quantization: keys with `k_mode`, values with `v_mode` ("channel" or "token")."""

    def __init__(self, bits: int, k_mode: str = "channel", v_mode: str = "token", group: int = 128):
        if bits not in (8, 4, 2):
            raise ValueError("bits must be 8, 4 or 2")
        self.bits, self.k_mode, self.v_mode, self.group = bits, k_mode, v_mode, int(group)
        self.name = f"int{bits}-k{k_mode[0]}v{v_mode[0]}"

    def _enc(self, xs, mode, prefix, arrays):
        L = len(xs)
        H, n, D = xs[0].shape[1:]
        qs, mns, sts = [], [], []
        for x in xs:
            q, mn, st = quantize(x[0], self.bits, mode, self.group)
            qs.append(q)
            mns.append(mn)
            sts.append(st)
        q = torch.stack(qs)
        arrays[prefix + "codes"] = pack_codes(q, self.bits)
        arrays[prefix + "min"] = torch.stack(mns)
        arrays[prefix + "step"] = torch.stack(sts)
        return q.numel() * self.bits // 8 + L * n_groups((H, n, D), mode, self.group) * 2 * FP16

    def encode(self, cache, start=0):
        arrays: dict = {}
        nb = self._enc(cache.keys, self.k_mode, "k_", arrays) + self._enc(cache.values, self.v_mode, "v_", arrays)
        L, H, n, D = len(cache.keys), *cache.keys[0].shape[1:]
        return Encoded(self.name, n, start, nb, arrays, {"shape": (L, H, n, D), "dtype": cache.dtype})

    def _dec(self, enc, prefix, mode):
        L, H, n, D = enc.meta["shape"]
        q = unpack_codes(enc.arrays[prefix + "codes"], self.bits, L * H * n * D).view(L, H, n, D)
        mn, st = enc.arrays[prefix + "min"], enc.arrays[prefix + "step"]
        dt = enc.meta["dtype"]
        return [dequantize(q[i], mn[i], st[i], mode, self.group).to(dt).unsqueeze(0) for i in range(L)]

    def decode(self, enc):
        return KVCache(self._dec(enc, "k_", self.k_mode), self._dec(enc, "v_", self.v_mode))


class LowRankBasis:
    """Per layer: mean and top-r principal directions of the 1,024-dim key vectors (rotated back to position
    0) and value vectors, from calibration caches. Stored as float16 (what encode and decode use)."""

    def __init__(self, k_mean, k_dirs, v_mean, v_dirs):
        self.k_mean, self.k_dirs, self.v_mean, self.v_dirs = k_mean, k_dirs, v_mean, v_dirs  # [L, F], [L, F, F] by variance

    @staticmethod
    def fit(caches_with_start, inv_freq: torch.Tensor, device="cuda") -> "LowRankBasis":
        """caches_with_start: iterable of (KVCache of one page, start position). Accumulates sums and Gram
        matrices in float64; returns all directions sorted by explained variance."""
        sk = gk = sv = gv = None
        count = 0
        for cache, start in caches_with_start:
            n = cache.n_tokens
            pos = torch.arange(start, start + n)
            ks, vs = [], []
            for kl, vl in zip(cache.keys, cache.values):
                k0 = rotate_positions(kl[0].float(), pos, inv_freq, -1)   # [H, n, D] at position 0
                ks.append(k0.permute(1, 0, 2).reshape(n, -1).double())    # [n, H*D]
                vs.append(vl[0].float().permute(1, 0, 2).reshape(n, -1).double())
            K, V = torch.stack(ks), torch.stack(vs)                           # [L, n, F]
            if sk is None:
                sk, sv = K.sum(1), V.sum(1)
                gk, gv = K.transpose(1, 2) @ K, V.transpose(1, 2) @ V
            else:
                sk += K.sum(1)
                sv += V.sum(1)
                gk += K.transpose(1, 2) @ K
                gv += V.transpose(1, 2) @ V
            count += n

        def pca(s, g):
            mu = s / count
            cov = g / count - mu[:, :, None] * mu[:, None, :]
            w, u = torch.linalg.eigh(cov)                 # ascending
            return mu, u.flip(-1), w.flip(-1)

        km, kd, kw = pca(sk, gk)
        vm, vd, vw = pca(sv, gv)
        b = LowRankBasis(km.half().to(device), kd.half().to(device), vm.half().to(device), vd.half().to(device))
        b.k_var, b.v_var, b.tokens = kw.float().cpu(), vw.float().cpu(), count
        return b

    def save(self, path) -> None:
        torch.save({"k_mean": self.k_mean.cpu(), "k_dirs": self.k_dirs.cpu(), "v_mean": self.v_mean.cpu(), "v_dirs": self.v_dirs.cpu(),
                    "k_var": getattr(self, "k_var", None), "v_var": getattr(self, "v_var", None), "tokens": getattr(self, "tokens", None)}, path)

    @staticmethod
    def load(path, device="cuda") -> "LowRankBasis":
        z = torch.load(path, map_location="cpu")
        b = LowRankBasis(z["k_mean"].to(device), z["k_dirs"].to(device), z["v_mean"].to(device), z["v_dirs"].to(device))
        b.k_var, b.v_var, b.tokens = z.get("k_var"), z.get("v_var"), z.get("tokens")
        return b


class LowRank(Codec):
    def __init__(self, basis: LowRankBasis, rank: int, bits: int, inv_freq: torch.Tensor, group: int = 128):
        self.basis, self.rank, self.bits, self.group = basis, int(rank), int(bits), int(group)
        self.inv_freq = inv_freq
        self.name = f"lowrank{rank}-int{bits}"
        L, F = basis.k_mean.shape
        self.shared_nbytes = 2 * L * (F * self.rank + F) * FP16
        self._k = basis.k_dirs[:, :, : self.rank].float()
        self._v = basis.v_dirs[:, :, : self.rank].float()
        self._km, self._vm = basis.k_mean.float(), basis.v_mean.float()

    def encode(self, cache, start=0):
        n = cache.n_tokens
        L, (H, _, D) = len(cache.keys), cache.keys[0].shape[1:]
        pos = torch.arange(start, start + n)
        arrays: dict = {}
        nb = 0
        for name, xs, U, mu in (("k_", cache.keys, self._k, self._km), ("v_", cache.values, self._v, self._vm)):
            cs = []
            for i, x in enumerate(xs):
                y = x[0].float()
                if name == "k_":
                    y = rotate_positions(y, pos, self.inv_freq, -1)
                y = y.permute(1, 0, 2).reshape(n, H * D)
                cs.append((y - mu[i]) @ U[i])                             # [n, r]
            c = torch.stack(cs).unsqueeze(1)                               # [L, 1, n, r]: "heads" = 1, channels = r
            qs, mns, sts = zip(*[quantize(c[i], self.bits, "channel", self.group) for i in range(L)])
            q = torch.stack(qs)
            arrays[name + "codes"] = pack_codes(q, self.bits)
            arrays[name + "min"], arrays[name + "step"] = torch.stack(mns), torch.stack(sts)
            nb += q.numel() * self.bits // 8 + L * n_groups((1, n, self.rank), "channel", self.group) * 2 * FP16
        return Encoded(self.name, n, start, nb, arrays, {"shape": (L, H, n, D), "dtype": cache.dtype})

    def decode(self, enc):
        L, H, n, D = enc.meta["shape"]
        pos = torch.arange(enc.start, enc.start + n)
        out = {}
        for name, U, mu in (("k_", self._k, self._km), ("v_", self._v, self._vm)):
            q = unpack_codes(enc.arrays[name + "codes"], self.bits, L * n * self.rank).view(L, 1, n, self.rank)
            mn, st = enc.arrays[name + "min"], enc.arrays[name + "step"]
            layers = []
            for i in range(L):
                c = dequantize(q[i], mn[i], st[i], "channel", self.group)[0]   # [n, r]
                y = (c @ U[i].T + mu[i]).view(n, H, D).permute(1, 0, 2)       # [H, n, D]
                if name == "k_":
                    y = rotate_positions(y, pos, self.inv_freq, +1)
                layers.append(y.to(enc.meta["dtype"]).unsqueeze(0))
            out[name] = layers
        return KVCache(out["k_"], out["v_"])


class Lossless(Codec):
    """An inner codec's stored arrays through a CPU lossless coder (optionally byte-shuffled first)."""

    def __init__(self, inner: Codec, coder_name: str, shuffle: bool = False):
        self.inner, self.coder_name, self.shuffle = inner, coder_name, bool(shuffle)
        self._c, self._d = coder(coder_name)
        self.name = f"{inner.name}{'-shuffle' if shuffle else ''}+{coder_name}"
        self.shared_nbytes = inner.shared_nbytes

    def encode(self, cache, start=0):
        enc = self.inner.encode(cache, start)
        raw = self.inner.to_bytes(enc)
        blob = self._c(shuffle_bytes(raw) if self.shuffle else raw)
        like = Encoded(enc.codec, enc.n_tokens, enc.start, enc.nbytes,
                       {k: torch.empty(0, dtype=a.dtype).new_empty(a.shape, device="meta") for k, a in enc.arrays.items()}, enc.meta)
        return Encoded(self.name, enc.n_tokens, start, len(blob), {"blob": blob}, {"like": like, "raw_bytes": len(raw)})

    def decode(self, enc, device="cuda"):
        raw = self._d(enc.arrays["blob"])
        if self.shuffle:
            raw = unshuffle_bytes(raw)
        inner = self.inner.from_bytes(raw, enc.meta["like"], device)
        return self.inner.decode(inner)


def make_codec(spec: str, basis: LowRankBasis | None = None, inv_freq: torch.Tensor | None = None, group: int = 128) -> Codec:
    """Codec from its name: bf16 | int{8,4,2}[-k{c,t}v{c,t}] | lowrank{r}-int{B} | <any of these>[-shuffle]+<coder>."""
    if "+" in spec:
        inner, cname = spec.rsplit("+", 1)
        shuffle = inner.endswith("-shuffle")
        if shuffle:
            inner = inner[: -len("-shuffle")]
        return Lossless(make_codec(inner, basis, inv_freq, group), cname, shuffle)
    if spec == "bf16":
        return BF16()
    if spec.startswith("int"):
        head, _, gran = spec.partition("-")
        modes = {"c": "channel", "t": "token"}
        k_mode, v_mode = ("channel", "token") if not gran else (modes[gran[1]], modes[gran[3]])
        return Quant(int(head[3:]), k_mode, v_mode, group)
    if spec.startswith("lowrank"):
        r, _, b = spec[len("lowrank"):].partition("-int")
        if basis is None or inv_freq is None:
            raise ValueError("a low-rank codec needs its fitted basis and the model's inv_freq")
        return LowRank(basis, int(r), int(b), inv_freq, group)
    raise ValueError(f"unknown codec {spec!r}")


def relative_error(ref: KVCache, got: KVCache) -> dict:
    """||got - ref||_F / ||ref||_F over all layers, for keys and for values."""
    def rel(a, b):
        num = sum(float((y.float() - x.float()).pow(2).sum()) for x, y in zip(a, b))
        den = sum(float(x.float().pow(2).sum()) for x in a)
        return math.sqrt(num / den)

    return {"k": rel(ref.keys, got.keys), "v": rel(ref.values, got.values)}


class TextCodec(Codec):
    """`text`: a page stored as its token ids (uint32), optionally through a lossless coder; decoding
    re-encodes the page after the shared prefix (pcm.store.kv.encode_page), so it costs a prefill of the page."""

    def __init__(self, model=None, prefix_cache: KVCache | None = None, coder_name: str | None = None, block: int = 2048):
        self.model, self.prefix, self.block = model, prefix_cache, block
        self.coder_name = coder_name
        self._c, self._d = coder(coder_name) if coder_name else (None, None)
        self.name = "text" + (f"+{coder_name}" if coder_name else "")

    def encode_ids(self, ids, start: int = 0) -> Encoded:
        raw = np.asarray(ids, dtype="<u4").tobytes()
        blob = self._c(raw) if self._c else raw
        return Encoded(self.name, len(raw) // 4, start, len(blob), {"blob": blob})

    def ids(self, enc: Encoded) -> torch.Tensor:
        raw = self._d(enc.arrays["blob"]) if self._d else enc.arrays["blob"]
        return torch.from_numpy(np.frombuffer(raw, dtype="<u4").astype(np.int64))

    def decode(self, enc: Encoded) -> KVCache:
        from pcm.store.kv import encode_page

        return encode_page(self.model, self.prefix, self.ids(enc), block=self.block)
