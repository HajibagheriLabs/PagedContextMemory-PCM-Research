"""Expansion of selected spans into the reader's window (specification, section 4.5).

TEXT mode: the token ids of the selected spans are joined in document order, with
a visible gap marker between spans that are not adjacent. Everything is done in
token space (span ids are slices of the context's own token ids; the marker is
tokenized on its own), the convention of every prompt in this project.

KV mode: the same token sequence, built from caches. The shared prefix P is encoded
once; every page, and the gap marker, is encoded on its own as a continuation of P
(it attends to P and to itself); the selected pages' caches are put in document
order after P, the marker's cache between pages that do not touch, each moved to its
new offset by re-basing its keys. The stitched context holds exactly the tokens of
the text-mode context, at the same positions; only what each token attended to when
it was encoded differs.
"""
from __future__ import annotations

from typing import Sequence

import torch


def expand_text(ctx_ids: torch.Tensor, spans: Sequence[Sequence[int]], gap_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """(token ids of the spans in document order with gap markers between them, number of markers).

    spans: token ranges [a, b) of the context; they must not overlap. Ranges that
    touch are joined without a marker.
    """
    ctx_ids = torch.as_tensor(ctx_ids, dtype=torch.long)
    ordered = sorted((int(a), int(b)) for a, b in spans if b > a)
    parts: list[torch.Tensor] = []
    gaps = 0
    prev_end = None
    for a, b in ordered:
        if a < 0 or b > len(ctx_ids):
            raise ValueError(f"span [{a}, {b}) lies outside the context of {len(ctx_ids)} tokens")
        if prev_end is not None:
            if a < prev_end:
                raise ValueError("spans overlap")
            if a > prev_end and gap_ids is not None and len(gap_ids):
                parts.append(torch.as_tensor(gap_ids, dtype=torch.long))
                gaps += 1
        parts.append(ctx_ids[a:b])
        prev_end = b
    if not parts:
        return ctx_ids[:0], 0
    return torch.cat(parts), gaps


def kv_layout(page_ranges: Sequence[Sequence[int]]) -> list[tuple[str, int, int]]:
    """The segments of the stitched KV context, in document order: ("page", a, b) for every page token
    range, and ("gap", 0, 0) between pages that do not touch, where text mode puts its gap marker."""
    out: list[tuple[str, int, int]] = []
    prev_end = None
    for a, b in sorted((int(a), int(b)) for a, b in page_ranges if b > a):
        if prev_end is not None:
            if a < prev_end:
                raise ValueError("pages overlap")
            if a > prev_end:
                out.append(("gap", 0, 0))
        out.append(("page", a, b))
        prev_end = b
    return out


def expand_kv(prefix_cache, segment_caches: Sequence, inv_freq: torch.Tensor, temperature: float = 1.0):
    """Prefix + segments (each encoded as a continuation of the prefix) stitched in the given order, keys
    re-based to their new offsets; with temperature T != 1 the keys of every context token (not the
    prefix) are divided by T. Returns (cache, context tokens)."""
    from pcm.store.kv import stitch

    cache = stitch(prefix_cache, segment_caches, inv_freq)
    p = prefix_cache.n_tokens if prefix_cache is not None else 0
    if temperature != 1.0:
        cache = cache.scale_keys(p, cache.n_tokens, 1.0 / float(temperature))
    return cache, cache.n_tokens - p
