"""Selection over index units: rankings, reciprocal-rank fusion, budget fill, oracle.

A ranking is an array of unit indices, best first. Every function is plain
arithmetic on arrays, so selection can be tested without a model.

- order_by_score: best first; equal scores go to the earlier unit.
- competition_ranks: rank 1 is best; equal scores share a rank (1 + the number of
  strictly better units). Used inside the fusion, so that units tied in one
  ranking (the many units with a BM25 score of 0) get no order from their position.
- rrf: reciprocal-rank fusion, the sum over rankings of 1 / (k + rank).
- fill_budget: the longest prefix of a ranking whose units fit the token budget.
- oracle_order: gold units first, then their neighbours by distance.
- spans_of / token_windows: kept units or evidence windows as merged token ranges
  in document order.
"""
from __future__ import annotations

import base64
from typing import Mapping, Sequence

import numpy as np


def order_by_score(scores: np.ndarray) -> np.ndarray:
    """Unit indices by decreasing score; ties in document order."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def competition_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank of every unit (1 = best); equal scores share the rank 1 + number of strictly better units."""
    s = np.asarray(scores, dtype=np.float64)
    sorted_desc = np.sort(-s)
    return np.searchsorted(sorted_desc, -s, side="left") + 1


def rrf(score_lists: Sequence[np.ndarray], k: float = 60.0) -> np.ndarray:
    """Fused score of every unit: sum over the rankings of 1 / (k + rank)."""
    if not score_lists:
        raise ValueError("nothing to fuse")
    out = np.zeros(len(score_lists[0]), dtype=np.float64)
    for s in score_lists:
        if len(s) != len(out):
            raise ValueError("rankings cover different numbers of units")
        out += 1.0 / (k + competition_ranks(s))
    return out


def rerank_head(order: np.ndarray, head_scores: np.ndarray) -> np.ndarray:
    """`order` with its first len(head_scores) entries re-ordered by decreasing head score
    (ties keep their previous order); the rest is unchanged."""
    order = np.asarray(order)
    m = len(head_scores)
    head = order[:m][np.argsort(-np.asarray(head_scores, dtype=np.float64), kind="stable")]
    return np.concatenate([head, order[m:]])


def fill_budget(order: np.ndarray, sizes: np.ndarray, budget: int) -> np.ndarray:
    """The longest prefix of `order` whose units' sizes sum to at most `budget` tokens."""
    if budget < 0:
        raise ValueError("budget must not be negative")
    total = np.cumsum(np.asarray(sizes)[np.asarray(order, dtype=np.int64)])
    return np.asarray(order[: int(np.searchsorted(total, budget, side="right"))], dtype=np.int64)


def spans_of(units: Sequence[int], bounds: np.ndarray) -> list[list[int]]:
    """Token ranges of the given units, in document order, adjacent units merged."""
    out: list[list[int]] = []
    for u in sorted(int(x) for x in set(int(v) for v in units)):
        a, b = int(bounds[u]), int(bounds[u + 1])
        if out and a == out[-1][1]:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def units_of_span(a: int, b: int, bounds: np.ndarray) -> list[int]:
    """Indices of the units that overlap tokens [a, b)."""
    if b <= a:
        return []
    first = int(np.searchsorted(bounds, a, side="right") - 1)
    last = int(np.searchsorted(bounds, b - 1, side="right") - 1)
    return list(range(max(0, first), min(len(bounds) - 2, last) + 1))


def evidence_units(evidence: Sequence[Mapping], bounds: np.ndarray) -> list[list[int]]:
    """For every evidence item, the units its token range [tok_start, tok_end) overlaps."""
    return [units_of_span(int(e["tok_start"]), int(e["tok_end"]), bounds) for e in evidence]


def item_ranks(order: np.ndarray, items: Sequence[Sequence[int]]) -> list[int]:
    """For every evidence item, how many units from the top of the ranking are needed to hold
    all of its units (1 = its units lead the ranking)."""
    pos = np.empty(len(order), dtype=np.int64)
    pos[np.asarray(order, dtype=np.int64)] = np.arange(1, len(order) + 1)
    return [int(max(pos[u] for u in units)) if len(units) else 0 for units in items]


def oracle_order(gold: Sequence[int], n_units: int, pad_units: int | None = None) -> np.ndarray:
    """Gold units in document order, then the other units by distance to the nearest gold
    unit (the earlier unit first on a tie). pad_units limits the neighbours to that distance."""
    g = np.asarray(sorted(set(int(u) for u in gold)), dtype=np.int64)
    if g.size == 0:
        raise ValueError("no gold unit")
    idx = np.arange(n_units)
    right = np.searchsorted(g, idx, side="left")
    d_right = np.where(right < g.size, g[np.minimum(right, g.size - 1)] - idx, n_units)
    d_left = np.where(right > 0, idx - g[np.maximum(right - 1, 0)], n_units)
    dist = np.minimum(d_left, d_right)
    order = np.lexsort((idx, dist))  # by distance, then by position; gold units have distance 0
    if pad_units is not None:
        order = order[dist[order] <= pad_units]
    return order


def token_windows(evidence: Sequence[Mapping], n_tokens: int, budget: int) -> tuple[list[list[int]], int]:
    """Every evidence item with the same margin of tokens on both sides, merged: the largest
    margin for which the total stays within `budget`. Returns (ranges, margin)."""
    def windows(margin: int) -> list[list[int]]:
        out: list[list[int]] = []
        for a, b in sorted((max(0, int(e["tok_start"]) - margin), min(n_tokens, int(e["tok_end"]) + margin)) for e in evidence):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    def total(margin: int) -> int:
        return sum(b - a for a, b in windows(margin))

    if total(0) > budget:
        raise ValueError("the evidence alone exceeds the budget")
    lo, hi = 0, n_tokens
    while lo < hi:  # largest margin whose total fits
        mid = (lo + hi + 1) // 2
        if total(mid) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return windows(lo), lo


def pack_units(units: Sequence[int], n_units: int) -> str:
    """A set of units as a base64 bitmap over the document's units (bit i set: unit i kept)."""
    bits = np.zeros(n_units, dtype=np.uint8)
    bits[np.asarray(list(units), dtype=np.int64)] = 1
    return base64.b64encode(np.packbits(bits).tobytes()).decode("ascii")


def unpack_units(packed: str, n_units: int) -> np.ndarray:
    """Inverse of pack_units: the sorted indices of the kept units."""
    bits = np.unpackbits(np.frombuffer(base64.b64decode(packed), dtype=np.uint8))[:n_units]
    return np.flatnonzero(bits)
