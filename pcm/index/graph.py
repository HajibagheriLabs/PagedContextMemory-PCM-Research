"""The relational graph over index units (specification, section 4.4) and its two traversals.

Nodes are index units. Three edge types, each switchable:

- SEQ: unit i <-> i +/- 1, weight 1.
- SEM: each unit's k = 5 nearest other units by cosine of their (normalised) embeddings,
  kept when the cosine is at least 0.5; weight = cosine.
- ENT: units that share entities; weight = IDF-weighted overlap (the sum of the IDF of
  the shared entities, IDF = log(units / units holding the entity)); each unit keeps
  its 20 heaviest such edges (the degree cap).

SEM and ENT are lists of out-edges per unit. A walk over several types averages the
types' row-normalised transition matrices over the types that have an edge at that
row, so that no type dominates by the scale of its weights (D-043).

Entities. Default extractor (no extra dependency), applied to each unit's text:
capitalised words that are not sentence-initial; terms in the top IDF decile of the
document (the 90th percentile of IDF over the document's term occurrences); numbers;
quoted strings; identifier-like strings (alphanumeric parts joined by '-' or '_', or
letters mixed with digits). Optional: the named entities of spaCy's small English
model. Entities are compared lower-cased.

Traversals:

- PPR: personalised PageRank from the top 16 units of the base ranking (uniform over
  them), restart 0.5, 10 power iterations on the GPU when there is one; the final
  ranking is the reciprocal-rank fusion of the base ranking and the PPR scores.
- Entity hops (H = 1 or 2): start from the same 16 units; at each hop collect the
  entities of the question and of the units selected so far, and add the 16 units
  not yet selected with the largest IDF-weighted overlap with them. The final
  ranking is the selected units in the order they were added, then every other unit
  in base order.
"""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pcm.index.bm25 import terms

EDGE_TYPES = ("seq", "sem", "ent")
EXTRACTORS = ("default", "spacy")

# ---------------------------------------------------------------- entities
_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z'’-]*")
_NUMBER = re.compile(r"\b\d+(?:[.,:]\d+)*\b")
_QUOTED = re.compile(r"\"([^\"\n]{1,64})\"|“([^”\n]{1,64})”")
_JOINED = re.compile(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b")
_MIXED = re.compile(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{2,}\b")
_OPENERS = set("\"'“‘([ \t\r")
_CLOSERS = set("\"'”’)]")


def sentence_initial(text: str, i: int) -> bool:
    """Whether the word starting at text[i] begins a sentence: nothing but spaces and opening quotes
    lie between it and the start of the text, a line break, or a sentence end (. ! ?, possibly
    followed by closing quotes)."""
    k = i - 1
    while k >= 0 and text[k] in _OPENERS:
        k -= 1
    if k < 0 or text[k] == "\n":
        return True
    while k >= 0 and text[k] in _CLOSERS:
        k -= 1
    return k >= 0 and text[k] in ".!?"


def _norm(s: str) -> str:
    s = s.strip().lower()
    for suffix in ("'s", "’s"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def surface_entities(text: str) -> set[str]:
    """The default extractor's entities of one text, except the top-IDF-decile terms (which need the
    document's statistics): capitalised words not at a sentence start, numbers, quoted strings,
    identifier-like strings."""
    out: set[str] = set()
    for m in _CAPITALISED.finditer(text):
        if not sentence_initial(text, m.start()):
            out.add(_norm(m.group()))
    out.update(m.group() for m in _NUMBER.finditer(text))
    for m in _QUOTED.finditer(text):
        out.add(_norm(m.group(1) or m.group(2)))
    out.update(_norm(m.group()) for m in _JOINED.finditer(text))
    out.update(_norm(m.group()) for m in _MIXED.finditer(text))
    out.discard("")
    return out


def rare_terms(unit_terms: Sequence[Sequence[str]]) -> tuple[set[str], dict[str, float]]:
    """Terms in the top IDF decile of the document, and the IDF of every term (IDF over units:
    log(units / units holding the term)). The decile is taken over term occurrences, so the threshold
    is the IDF above which the rarest 10% of the document's running text lies."""
    n = len(unit_terms)
    df: dict[str, int] = {}
    for ts in unit_terms:
        for t in set(ts):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(n / d) for t, d in df.items()}
    occ = np.fromiter((idf[t] for ts in unit_terms for t in ts), dtype=np.float64)
    if occ.size == 0:
        return set(), idf
    threshold = float(np.quantile(occ, 0.9))
    return {t for t, v in idf.items() if v >= threshold and v > 0}, idf


class Entities:
    """Entities of a document's units: a vocabulary, the unit x entity incidence (CSR), and IDF."""

    def __init__(self, vocab: list[str], indptr: np.ndarray, indices: np.ndarray, idf: np.ndarray, extra: dict | None = None):
        self.vocab = list(vocab)
        self.index = {e: i for i, e in enumerate(self.vocab)}
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.idf = np.asarray(idf, dtype=np.float64)
        self.extra = extra or {}  # what the question extractor needs (default: the rare-term set)

    @property
    def n_units(self) -> int:
        return len(self.indptr) - 1

    def of_unit(self, u: int) -> np.ndarray:
        return self.indices[self.indptr[u]: self.indptr[u + 1]]

    def matrix(self):
        """Binary unit x entity matrix (scipy CSR)."""
        from scipy.sparse import csr_matrix

        return csr_matrix((np.ones(len(self.indices)), self.indices, self.indptr), shape=(self.n_units, len(self.vocab)))

    def ids(self, names: Iterable[str]) -> list[int]:
        return sorted({self.index[n] for n in names if n in self.index})

    @classmethod
    def from_sets(cls, sets: Sequence[set[str]], extra: dict | None = None) -> "Entities":
        vocab = sorted(set().union(*sets)) if sets else []
        index = {e: i for i, e in enumerate(vocab)}
        indptr = [0]
        indices: list[int] = []
        for s in sets:
            indices.extend(sorted(index[e] for e in s))
            indptr.append(len(indices))
        df = np.bincount(np.asarray(indices, dtype=np.int64), minlength=len(vocab)).astype(np.float64)
        n = len(sets)
        idf = np.log(n / np.maximum(df, 1.0)) if len(vocab) else np.zeros(0)
        return cls(vocab, np.asarray(indptr), np.asarray(indices), idf, extra)


def default_entities(texts: Sequence[str]) -> Entities:
    """The default extractor over a document's unit texts."""
    unit_terms = [terms(t) for t in texts]
    rare, _ = rare_terms(unit_terms)
    sets = [surface_entities(t) | (set(ts) & rare) for t, ts in zip(texts, unit_terms)]
    return Entities.from_sets(sets, {"rare": sorted(rare)})


_SPACY = {}


def spacy_model():
    """spaCy's small English model with the NER pipe only (loaded once)."""
    if "nlp" not in _SPACY:
        import spacy

        _SPACY["nlp"] = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "attribute_ruler", "tagger", "senter"])
        _SPACY["nlp"].max_length = 10**7
    return _SPACY["nlp"]


def spacy_entities_of(texts: Sequence[str]) -> list[set[str]]:
    nlp = spacy_model()
    return [{_norm(e.text) for e in doc.ents} - {""} for doc in nlp.pipe(texts, batch_size=64)]


def extract(texts: Sequence[str], extractor: str = "default") -> Entities:
    if extractor == "default":
        return default_entities(texts)
    if extractor == "spacy":
        return Entities.from_sets(spacy_entities_of(texts))
    raise ValueError(f"unknown entity extractor {extractor!r}; known: {EXTRACTORS}")


def question_entities(question: str, ents: Entities, extractor: str = "default") -> list[int]:
    """Entity ids (in the document's vocabulary) found in the question by the same extractor; for the
    default extractor the top-IDF-decile terms are the document's."""
    if extractor == "default":
        names = surface_entities(question) | (set(terms(question)) & set(ents.extra.get("rare", [])))
    else:
        names = spacy_entities_of([question])[0]
    return ents.ids(names)


# ---------------------------------------------------------------- edges
def ent_edges(ents: Entities, cap: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ENT out-edges (rows, cols, weights): for every unit, the `cap` other units with the largest
    IDF-weighted entity overlap (> 0); ties go to the nearer unit, then the earlier."""
    n = ents.n_units
    if n == 0 or not len(ents.vocab):
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    from scipy.sparse import diags

    b = ents.matrix()
    w = (b @ diags(ents.idf) @ b.T).tocsr()
    w.setdiag(0)
    w.eliminate_zeros()
    rows, cols, vals = [], [], []
    for i in range(n):
        a, z = w.indptr[i], w.indptr[i + 1]
        if a == z:
            continue
        c, v = w.indices[a:z], w.data[a:z]
        top = np.lexsort((c, np.abs(c - i), -v))[:cap]
        rows.append(np.full(len(top), i))
        cols.append(c[top])
        vals.append(v[top])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64), np.concatenate(vals).astype(np.float64)


def sem_edges(emb: np.ndarray, k: int = 5, min_cos: float = 0.5, device: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SEM out-edges (rows, cols, cosines): every unit's k nearest other units with cosine >= min_cos.
    Computed in float32 on the GPU when there is one; ties by the earlier unit."""
    import torch

    n = len(emb)
    if n < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    e = torch.as_tensor(np.asarray(emb, dtype=np.float32), device=dev)
    rows, cols, vals = [], [], []
    for a in range(0, n, 4096):  # blocks of rows keep the similarity matrix small at 1M tokens
        s = e[a: a + 4096] @ e.T
        idx = torch.arange(a, min(a + 4096, n), device=dev)
        s[idx - a, idx] = -float("inf")
        v, c = torch.topk(s, min(k, n - 1), dim=1, sorted=True)
        v, c = v.double().cpu().numpy(), c.cpu().numpy()
        keep = v >= min_cos
        r = np.repeat(np.arange(a, a + len(v))[:, None], v.shape[1], axis=1)
        rows.append(r[keep])
        cols.append(c[keep])
        vals.append(v[keep])
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols).astype(np.int64), np.concatenate(vals)


def seq_edges(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.arange(n - 1, dtype=np.int64)
    return np.concatenate([i, i + 1]), np.concatenate([i + 1, i]), np.ones(2 * (n - 1))


@dataclass
class Graph:
    """Edges of one document. Each entry of `edges` is (rows, cols, weights) of out-edges."""

    n: int
    edges: dict

    def transition(self, types: Sequence[str]):
        """Row-stochastic transition matrix (scipy CSR) averaging the given edge types (D-043); rows
        with no edge of any of the types are all zero (dangling)."""
        from scipy.sparse import csr_matrix, diags

        total = csr_matrix((self.n, self.n))
        count = np.zeros(self.n)
        for t in types:
            if t not in EDGE_TYPES:
                raise ValueError(f"unknown edge type {t!r}; known: {EDGE_TYPES}")
            r, c, w = seq_edges(self.n) if t == "seq" else self.edges[t]
            if len(r) == 0:
                continue
            m = csr_matrix((w, (r, c)), shape=(self.n, self.n))
            s = np.asarray(m.sum(axis=1)).ravel()
            has = s > 0
            m = diags(np.where(has, 1.0 / np.where(has, s, 1.0), 0.0)) @ m
            total = total + m
            count += has
        return (diags(np.where(count > 0, 1.0 / np.maximum(count, 1), 0.0)) @ total).tocsr()


# ---------------------------------------------------------------- traversals
def ppr(transition, seeds: Sequence[int], restart: float = 0.5, iterations: int = 10, device: str | None = None) -> np.ndarray:
    """Personalised PageRank scores: p <- r s + (1 - r) (P^T p + dangling mass x s), from p = s,
    with s uniform over the seeds. Sparse power iteration in float64 (on the GPU when there is one);
    scores are rounded to 12 significant digits of the largest, so that ties do not depend on the order
    of floating-point sums."""
    import torch

    n = transition.shape[0]
    s = np.zeros(n)
    seeds = sorted(set(int(x) for x in seeds))
    if not seeds:
        raise ValueError("PPR needs at least one seed")
    s[seeds] = 1.0 / len(seeds)
    dangling = np.asarray(transition.sum(axis=1)).ravel() == 0
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pt = transition.T.tocsr()
    m = torch.sparse_csr_tensor(torch.as_tensor(pt.indptr, dtype=torch.int64), torch.as_tensor(pt.indices, dtype=torch.int64),
                                torch.as_tensor(pt.data, dtype=torch.float64), size=(n, n), device=dev,
                                check_invariants=False)
    st = torch.as_tensor(s, dtype=torch.float64, device=dev)
    dg = torch.as_tensor(dangling, device=dev)
    p = st.clone()
    for _ in range(int(iterations)):
        p = restart * st + (1.0 - restart) * ((m @ p.unsqueeze(1)).squeeze(1) + p[dg].sum() * st)
    out = p.cpu().numpy()
    top = out.max()
    return np.round(out / top, 12) * top if top > 0 else out


def entity_hops(base_order: np.ndarray, ents: Entities, q_ents: Sequence[int], hops: int, seeds: int = 16, hop_k: int = 16) -> tuple[np.ndarray, list[int]]:
    """Ranking after H entity hops from the top `seeds` units of `base_order` (see the module text).
    Returns (ranking, number of units added at each hop)."""
    base_order = np.asarray(base_order, dtype=np.int64)
    n = len(base_order)
    pos = np.empty(n, dtype=np.int64)
    pos[base_order] = np.arange(n)
    chosen = list(base_order[:seeds])
    inside = np.zeros(n, dtype=bool)
    inside[chosen] = True
    known = set(int(e) for e in q_ents)
    b = ents.matrix() if len(ents.vocab) else None
    added = []
    for _ in range(int(hops)):
        for u in chosen:
            known.update(int(e) for e in ents.of_unit(int(u)))
        if b is None or not known:
            added.append(0)
            continue
        w = np.zeros(len(ents.vocab))
        idx = np.fromiter(known, dtype=np.int64)
        w[idx] = ents.idf[idx]
        overlap = np.asarray(b @ w).ravel()
        overlap[inside] = 0.0
        cand = np.lexsort((pos, -overlap))  # largest overlap first, then base rank
        new = [int(u) for u in cand[:hop_k] if overlap[u] > 0]
        chosen.extend(new)
        inside[new] = True
        added.append(len(new))
    rest = base_order[~inside[base_order]]
    return np.concatenate([np.asarray(chosen, dtype=np.int64), rest]), added


# ---------------------------------------------------------------- per-document cache
class GraphCache:
    """Entities and edges of a document, stored once (section 4.7 rule 10) beside its embeddings:

        <index cache dir>/graph/<name>/<hh>/<hash>.npz

    `name` says what the file holds: "ent-<extractor>-c<cap>" (entities and ENT edges) or
    "sem-k<k>-m<min_cos>" (SEM edges, which depend on the embeddings). Written atomically."""

    def __init__(self, root: Path):
        self.dir = Path(root) / "graph"

    def path(self, name: str, key: str) -> Path:
        return self.dir / name / key[:2] / f"{key}.npz"

    def save(self, name: str, key: str, arrays: dict) -> int:
        from pcm.runlog import atomic_write_bytes

        buf = io.BytesIO()
        np.savez(buf, **arrays)
        data = buf.getvalue()
        atomic_write_bytes(self.path(name, key), data)
        return len(data)

    def load(self, name: str, key: str) -> dict | None:
        p = self.path(name, key)
        if not p.is_file():
            return None
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def ent_name(extractor: str, cap: int) -> str:
    return f"ent-{extractor}-c{cap}"


def sem_name(k: int, min_cos: float) -> str:
    return f"sem-k{k}-m{min_cos:g}"


def pack_entities(ents: Entities, edges: tuple) -> dict:
    r, c, w = edges
    return {"vocab": np.asarray(ents.vocab, dtype=str), "indptr": ents.indptr, "indices": ents.indices, "idf": ents.idf,
            "rare": np.asarray(ents.extra.get("rare", []), dtype=str), "rows": r, "cols": c, "weights": w}


def unpack_entities(z: dict) -> tuple[Entities, tuple]:
    ents = Entities([str(x) for x in z["vocab"]], z["indptr"], z["indices"], z["idf"], {"rare": [str(x) for x in z["rare"]]})
    return ents, (z["rows"], z["cols"], z["weights"])
