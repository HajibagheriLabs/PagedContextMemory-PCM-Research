"""Expansion of selected spans into the reader's window (specification, section 4.5).

TEXT mode: the token ids of the selected spans are joined in document order, with
a visible gap marker between spans that are not adjacent. Everything is done in
token space (span ids are slices of the context's own token ids; the marker is
tokenized on its own), the convention of every prompt in this project.
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
