"""Index units and pages (specification, section 4.1).

A document of n tokens is cut into consecutive, non-overlapping units of about
`unit` tokens. Every nominal boundary (a multiple of `unit`) moves to the nearest
sentence end within `snap` tokens, the earlier one on a tie; with no sentence end
in reach it stays where it is. A last piece shorter than `snap` tokens joins the
unit before it. Units are described by their boundaries: an integer array
[0, b1, ..., n], unit i covering tokens [b[i], b[i + 1]).

A page is a run of consecutive units: with page size P tokens, page p holds units
[p k, (p + 1) k) where k = P / unit, counted from the first unit; the last page may
hold fewer. Pages therefore inherit the units' sentence snapping, and a page of
P tokens is P tokens give or take the snapping of its two edges.

Token indices refer to the context tokenized on its own with the reader's
tokenizer, the convention of every offset in this project.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pcm.tasks.common import SENTENCE_END, tokenize_with_offsets


def sentence_end_tokens(text: str, starts: Sequence[int]) -> np.ndarray:
    """Token indices at which a new sentence begins: for each sentence end in `text`, the first
    token that starts at or after it. Sorted, without duplicates."""
    out = {bisect_left(starts, m.end()) for m in SENTENCE_END.finditer(text)}
    return np.fromiter(sorted(out), dtype=np.int64, count=len(out))


def unit_bounds(n_tokens: int, sentence_bounds: Sequence[int] | np.ndarray = (), unit: int = 128, snap: int = 32) -> np.ndarray:
    """Boundaries of the index units of a document of n_tokens tokens."""
    if n_tokens < 0:
        raise ValueError("n_tokens must not be negative")
    if unit < 1 or snap < 0 or unit <= 2 * snap:
        raise ValueError("need unit > 2 x snap >= 0, so that snapped boundaries stay in order")
    if n_tokens == 0:
        return np.zeros(1, dtype=np.int64)
    sb = np.asarray(sentence_bounds, dtype=np.int64)
    bounds = [0]
    for nominal in range(unit, n_tokens, unit):
        b = nominal
        if snap and sb.size:
            cand = sb[np.searchsorted(sb, nominal - snap, "left"): np.searchsorted(sb, nominal + snap, "right")]
            cand = cand[(cand > bounds[-1]) & (cand < n_tokens)]
            if cand.size:
                b = int(cand[np.argmin(np.abs(cand - nominal))])  # argmin returns the first: the earlier one on a tie
        bounds.append(b)
    if len(bounds) > 1 and n_tokens - bounds[-1] < snap:
        bounds.pop()  # a last piece shorter than `snap` tokens joins the unit before it
    bounds.append(n_tokens)
    return np.asarray(bounds, dtype=np.int64)


@dataclass
class Units:
    """A tokenized document and its index units."""

    ids: np.ndarray       # token ids of the context, int32
    starts: np.ndarray    # character start of every token
    ends: np.ndarray      # character end of every token
    bounds: np.ndarray    # unit boundaries in tokens, length n_units + 1

    @property
    def n_units(self) -> int:
        return len(self.bounds) - 1

    @property
    def n_tokens(self) -> int:
        return int(self.bounds[-1])

    @property
    def sizes(self) -> np.ndarray:
        return np.diff(self.bounds)

    def char_span(self, i: int) -> tuple[int, int]:
        """Characters of unit i: from its first token's start to its last token's end."""
        return int(self.starts[self.bounds[i]]), int(self.ends[self.bounds[i + 1] - 1])

    def texts(self, context: str) -> list[str]:
        return [context[a:b] for a, b in (self.char_span(i) for i in range(self.n_units))]


def cut_document(tokenizer, context: str, unit: int = 128, snap: int = 32) -> Units:
    """Tokenize `context` and cut it into index units."""
    ids, starts, ends = tokenize_with_offsets(tokenizer, context)
    bounds = unit_bounds(len(ids), sentence_end_tokens(context, starts), unit, snap)
    return Units(np.asarray(ids, dtype=np.int32), np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64), bounds)


def units_per_page(page_tokens: int, unit: int = 128) -> int:
    """How many index units make a page of `page_tokens` tokens (a whole multiple of the unit)."""
    if page_tokens < unit or page_tokens % unit:
        raise ValueError(f"a page of {page_tokens} tokens is not a whole number of {unit}-token units")
    return page_tokens // unit


def page_of_unit(n_units: int, per_page: int) -> np.ndarray:
    """Page index of every unit."""
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    return np.arange(n_units, dtype=np.int64) // per_page


def page_bounds(bounds: np.ndarray, per_page: int) -> np.ndarray:
    """Token boundaries of the pages made of `per_page` consecutive units (unit boundaries `bounds`)."""
    n_units = len(bounds) - 1
    if per_page < 1:
        raise ValueError("per_page must be at least 1")
    idx = list(range(0, n_units, per_page)) + [n_units]
    return np.asarray(bounds, dtype=np.int64)[idx]
