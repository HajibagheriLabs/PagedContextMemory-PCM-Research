"""BM25 (Okapi) over the index units of one document.

Terms are lower-cased runs of word characters: no stemming, no stop list. The
formula and its defaults (k1 = 1.5, b = 0.75, negative IDF replaced by
epsilon x mean IDF with epsilon = 0.25) are those of the `rank_bm25` package's
BM25Okapi; tests/test_index.py checks the scores against it.

Statistics are kept as postings (unit ids and term frequencies per term), so
scoring a query touches only the units that contain one of its terms.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

WORD = re.compile(r"\w+")


def terms(text: str) -> list[str]:
    return WORD.findall(text.lower())


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = float(k1), float(b), float(epsilon)
        self.n_docs = len(docs)
        self.doc_len = np.asarray([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        ids: dict[str, list[int]] = {}
        tfs: dict[str, list[int]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                ids.setdefault(term, []).append(i)
                tfs.setdefault(term, []).append(tf)
        self.postings = {t: (np.asarray(ids[t], dtype=np.int64), np.asarray(tfs[t], dtype=np.float64)) for t in ids}
        idf = {t: math.log(self.n_docs - len(p[0]) + 0.5) - math.log(len(p[0]) + 0.5) for t, p in self.postings.items()}
        mean_idf = sum(idf.values()) / len(idf) if idf else 0.0
        self.idf = {t: (v if v >= 0 else self.epsilon * mean_idf) for t, v in idf.items()}
        # Length normalisation of every unit: k1 x (1 - b + b x length / mean length).
        self._norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl) if self.avgdl else np.zeros(self.n_docs)

    @classmethod
    def from_texts(cls, texts: Sequence[str], **kw) -> "BM25":
        return cls([terms(t) for t in texts], **kw)

    def scores(self, query: Sequence[str] | str) -> np.ndarray:
        """BM25 score of every unit for the query (a string, or its terms; repeated terms count each time)."""
        q = terms(query) if isinstance(query, str) else list(query)
        out = np.zeros(self.n_docs, dtype=np.float64)
        for t in q:
            p = self.postings.get(t)
            if p is None:
                continue
            ids, tf = p
            out[ids] += self.idf[t] * tf * (self.k1 + 1) / (tf + self._norm[ids])
        return out
