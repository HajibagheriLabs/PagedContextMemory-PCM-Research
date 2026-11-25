"""KV cache adapter, key re-basing and page encoding.

This is the only module that touches the library's cache objects. Everything
else works with KVCache, a plain container of per-layer key and value tensors
of shape [1, kv_heads, tokens, head_dim].

What the installed library caches (transformers 5.19, Qwen3Attention.forward):
keys after k_norm and after the rotary transform, values with no positional
component. A cached key at position p is therefore R(p) k. Rotations about the
same planes compose, R(d) R(p) = R(p + d), so applying the rotary transform once
more with the constant position d moves a cached page by d positions. Values
are left alone. Test T2 checks this against the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class KVCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("keys and values must have the same number of layers")

    # ---- shape and size ----
    @property
    def n_layers(self) -> int:
        return len(self.keys)

    @property
    def n_tokens(self) -> int:
        return self.keys[0].shape[-2] if self.keys else 0

    def __len__(self) -> int:
        return self.n_tokens

    @property
    def dtype(self) -> torch.dtype:
        return self.keys[0].dtype

    @property
    def device(self) -> torch.device:
        return self.keys[0].device

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (*self.keys, *self.values))

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys[i], self.values[i]

    # ---- token-range operations ----
    def slice(self, start: int, stop: int) -> "KVCache":
        """Tokens [start, stop) as views of this cache. Call clone() to detach the memory."""
        if not 0 <= start <= stop <= self.n_tokens:
            raise ValueError(f"bad token range [{start}, {stop}) for a cache of {self.n_tokens} tokens")
        return KVCache(
            [k[:, :, start:stop] for k in self.keys],
            [v[:, :, start:stop] for v in self.values],
        )

    def clone(self) -> "KVCache":
        return KVCache(
            [k.contiguous().clone() for k in self.keys],
            [v.contiguous().clone() for v in self.values],
        )

    @staticmethod
    def concat(caches: Sequence["KVCache"]) -> "KVCache":
        """Concatenate along the token axis, in the given order."""
        caches = list(caches)
        if not caches:
            raise ValueError("nothing to concatenate")
        n = caches[0].n_layers
        if any(c.n_layers != n for c in caches):
            raise ValueError("caches have different numbers of layers")
        return KVCache(
            [torch.cat([c.keys[i] for c in caches], dim=-2) for i in range(n)],
            [torch.cat([c.values[i] for c in caches], dim=-2) for i in range(n)],
        )

    def to(self, device=None, dtype=None) -> "KVCache":
        return KVCache(
            [k.to(device=device, dtype=dtype) for k in self.keys],
            [v.to(device=device, dtype=dtype) for v in self.values],
        )

    def rebase(self, delta: int, inv_freq: torch.Tensor) -> "KVCache":
        """Move every token by `delta` positions: keys rotated, values shared unchanged."""
        if delta == 0:
            return KVCache(list(self.keys), list(self.values))
        return KVCache([rebase_keys(k, delta, inv_freq) for k in self.keys], list(self.values))

    def scale_keys(self, start: int, stop: int, factor: float) -> "KVCache":
        """Keys of tokens [start, stop) multiplied by `factor` (in float32, stored back in the cache's dtype);
        everything else shared unchanged. With factor 1 / T this divides those tokens' attention scores by T
        (APE-style temperature, section 4.5)."""
        if not 0 <= start <= stop <= self.n_tokens:
            raise ValueError(f"bad token range [{start}, {stop}) for a cache of {self.n_tokens} tokens")
        if factor == 1.0 or start == stop:
            return KVCache(list(self.keys), list(self.values))
        keys = []
        for k in self.keys:
            k2 = k.clone()
            k2[:, :, start:stop] = (k[:, :, start:stop].float() * float(factor)).to(k.dtype)
            keys.append(k2)
        return KVCache(keys, list(self.values))

    # ---- library cache conversion ----
    def to_library(self, config=None, capacity: int | None = None):
        """A library cache object holding this cache. Forward passes on it never modify this KVCache.

        Without `capacity` the layers reference these tensors (zero-copy) and grow
        by concatenation into new tensors. With `capacity` (total tokens the cache
        will hold) each layer is one preallocated buffer into which this cache is
        copied once and later tokens are written in place.
        """
        lib = new_library_cache(config, self.n_layers, capacity)
        if capacity is not None:
            if capacity < self.n_tokens:
                raise ValueError("capacity is smaller than the cache")
            for layer, k, v in zip(lib.layers, self.keys, self.values):
                layer.update(k, v)
            return lib
        for layer, k, v in zip(lib.layers, self.keys, self.values):
            layer.dtype, layer.device = k.dtype, k.device
            layer.keys, layer.values = k, v
            layer.is_initialized = True
        return lib

    @classmethod
    def from_library(cls, lib) -> "KVCache":
        keys, values = [], []
        for layer in lib.layers:
            if not getattr(layer, "is_initialized", False) or layer.keys is None:
                raise ValueError("library cache has an empty layer")
            keys.append(layer.keys)
            values.append(layer.values)
        return cls(keys, values)


def _buffer_layer_class():
    """A cache layer that writes into one preallocated buffer instead of concatenating.

    The library's dynamic layer builds a new, longer tensor on every update. Over a
    long chunked prefill that copies the whole cache once per block and leaves the
    allocator holding many freed blocks of growing sizes. This layer allocates
    `capacity` tokens once and hands the attention a view of the filled part.
    """
    from transformers.cache_utils import DynamicLayer

    class BufferLayer(DynamicLayer):
        def __init__(self, capacity: int):
            super().__init__()
            self.capacity = int(capacity)
            self.length = 0
            self._k = self._v = None

        def _allocate(self, like_k: torch.Tensor, like_v: torch.Tensor, capacity: int):
            shape_k = (*like_k.shape[:-2], capacity, like_k.shape[-1])
            shape_v = (*like_v.shape[:-2], capacity, like_v.shape[-1])
            k = torch.empty(shape_k, dtype=like_k.dtype, device=like_k.device)
            v = torch.empty(shape_v, dtype=like_v.dtype, device=like_v.device)
            return k, v

        def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
            self.dtype, self.device = key_states.dtype, key_states.device
            self._k, self._v = self._allocate(key_states, value_states, self.capacity)
            self.keys, self.values = self._k[..., :0, :], self._v[..., :0, :]
            self.is_initialized = True

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            end = self.length + key_states.shape[-2]
            if end > self.capacity:  # grow geometrically; only reached if the caller under-declared
                capacity = max(end, self.capacity + self.capacity // 2)
                k, v = self._allocate(self._k, self._v, capacity)
                k[..., : self.length, :] = self._k[..., : self.length, :]
                v[..., : self.length, :] = self._v[..., : self.length, :]
                self._k, self._v, self.capacity = k, v, capacity
            self._k[..., self.length : end, :] = key_states
            self._v[..., self.length : end, :] = value_states
            self.length = end
            self.keys, self.values = self._k[..., :end, :], self._v[..., :end, :]
            return self.keys, self.values

        def get_seq_length(self) -> int:
            return self.length

    return BufferLayer


def new_library_cache(config=None, n_layers: int | None = None, capacity: int | None = None):
    """An empty library cache with one layer object per model layer.

    With `capacity`, layers are preallocated buffers of that many tokens.
    """
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer

    count = n_layers
    if config is not None:
        probe = DynamicCache(config=config)
        if probe.layers:
            if any(type(layer) is not DynamicLayer for layer in probe.layers):
                raise TypeError("only plain full-attention cache layers are supported")
            if n_layers is not None and len(probe.layers) != n_layers:
                raise ValueError(f"cache has {n_layers} layers, model config has {len(probe.layers)}")
            count = len(probe.layers)
            if capacity is None:
                return probe
    lib = DynamicCache()
    if count:
        if capacity is None:
            lib.layers = [DynamicLayer() for _ in range(count)]
        else:
            layer_cls = _buffer_layer_class()
            lib.layers = [layer_cls(capacity) for _ in range(count)]
        lib.layer_class_to_replicate = None
    elif capacity is not None:
        raise ValueError("a preallocated cache needs the number of layers (pass config or n_layers)")
    return lib


def library_cache_length(lib) -> int:
    return lib.get_seq_length()


# ---- rotary transform ----
def rotary_module(model):
    """The model's rotary embedding module."""
    base = getattr(model, "model", model)
    rot = getattr(base, "rotary_emb", None)
    if rot is None:
        raise AttributeError("model has no rotary_emb module")
    return rot


def rope_inv_freq(model) -> torch.Tensor:
    """The model's own inverse frequencies (float64, CPU). Never recomputed from the config."""
    return rotary_module(model).inv_freq.detach().to("cpu", torch.float64)


def rope_attention_scaling(model) -> float:
    return float(getattr(rotary_module(model), "attention_scaling", 1.0))


def rebase_keys(K: torch.Tensor, delta: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply the rotary transform for the constant position `delta` to cached keys.

    K: [..., head_dim] keys that already carry their rotation. Returns keys of the
    same dtype positioned `delta` tokens later (earlier for a negative delta).
    The angle is computed in float64; the rotation is applied in at least float32.
    Uses the library's layout: the two halves of head_dim are the paired coordinates.
    """
    if delta == 0:
        return K
    angle = float(delta) * inv_freq.to(torch.float64)
    work = torch.float64 if K.dtype == torch.float64 else torch.float32
    cos = torch.cat((angle.cos(), angle.cos())).to(device=K.device, dtype=work)
    sin = torch.cat((angle.sin(), angle.sin())).to(device=K.device, dtype=work)
    k = K.to(work)
    half = k.shape[-1] // 2
    rotated_half = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    return (k * cos + rotated_half * sin).to(K.dtype)


# ---- pages ----
def encode_page(model, prefix_cache: KVCache | None, page_ids: torch.Tensor, block: int = 2048) -> KVCache:
    """Encode one page as a continuation of the shared prefix; return the page-only cache.

    The page attends to the prefix and to itself. Its tokens are encoded at
    positions len(prefix) .. len(prefix) + len(page) - 1.
    """
    from pcm.prefill import prefill_in_blocks

    p = prefix_cache.n_tokens if prefix_cache is not None else 0
    full = prefill_in_blocks(model, page_ids, block=block, past=prefix_cache)
    return full.slice(p, full.n_tokens).clone()


def stitch(
    prefix_cache: KVCache | None,
    page_caches: Sequence[KVCache],
    inv_freq: torch.Tensor,
    encoded_at: int | None = None,
) -> KVCache:
    """Prefix followed by the pages in the given order, each page re-based to its new offset.

    Every page cache must have been encoded starting at position `encoded_at`
    (default: the prefix length, which is what encode_page produces). Page j is
    moved by (prefix length + tokens of the pages before it) - encoded_at.
    """
    p = prefix_cache.n_tokens if prefix_cache is not None else 0
    origin = p if encoded_at is None else encoded_at
    pages = list(page_caches)
    if not pages:
        return KVCache.concat([prefix_cache])
    # One rotation per layer over all pages at once: each token gets its page's delta. The angles are the
    # same float64 products as in rebase_keys, so the result equals re-basing page by page, bit for bit.
    deltas, start = [], p
    for page in pages:
        deltas.append(torch.full((page.n_tokens,), float(start - origin), dtype=torch.float64))
        start += page.n_tokens
    first = pages[0].keys[0]
    work = torch.float64 if first.dtype == torch.float64 else torch.float32
    angle = torch.cat(deltas)[:, None] * inv_freq.to(torch.float64).cpu()[None, :]
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1).to(device=first.device, dtype=work)
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1).to(device=first.device, dtype=work)
    half = first.shape[-1] // 2
    keys, values = [], []
    for i in range(pages[0].n_layers):
        k = torch.cat([pg.keys[i] for pg in pages], dim=-2)
        w = k.to(work)
        k = (w * cos + torch.cat((-w[..., half:], w[..., :half]), dim=-1) * sin).to(k.dtype)
        v = torch.cat([pg.values[i] for pg in pages], dim=-2)
        if prefix_cache is not None:
            k = torch.cat([prefix_cache.keys[i], k], dim=-2)
            v = torch.cat([prefix_cache.values[i], v], dim=-2)
        keys.append(k)
        values.append(v)
    return KVCache(keys, values)
