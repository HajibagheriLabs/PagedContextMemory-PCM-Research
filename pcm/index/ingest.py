"""Per-document index cache (specification, section 4.7 rule 10).

Ingest of a document = cutting it into index units and embedding every unit. The
result is stored once per document, keyed by a hash of the document's token ids,
under

    <root>/<tokenizer tag>/<embedder tag>/u<unit>s<snap>/<hh>/<hash>.npz

with the unit boundaries (int32) and the unit embeddings (float16, normalised).
Each file is written under a temporary name and renamed, so a file that exists is
complete. One line per ingested document goes to ingest.jsonl beside the files
(tokens, units, seconds, tokens per second, code commit).

BM25 statistics are not stored: they are rebuilt from the unit texts at query
time, which takes a fraction of a second per document.
"""
from __future__ import annotations

import hashlib
import io
import time
from pathlib import Path

import numpy as np

from pcm.chunking import Units
from pcm.runlog import JsonlLog, atomic_write_bytes


def doc_hash(ids: np.ndarray) -> str:
    """Hash of a document's token ids (int32, little endian)."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i4").tobytes()).hexdigest()[:32]


class IndexCache:
    def __init__(self, root: Path, tokenizer_tag: str, embedder_tag: str, unit: int, snap: int):
        self.dir = Path(root) / tokenizer_tag / embedder_tag / f"u{unit}s{snap}"
        self.log = JsonlLog(self.dir / "ingest.jsonl", id_field="doc")

    def path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def load(self, key: str, bounds: np.ndarray | None = None) -> np.ndarray | None:
        """The stored embeddings (float32) of a document, or None. With `bounds`, the stored
        boundaries must equal them: the cache then cannot serve another way of cutting."""
        p = self.path(key)
        if not p.is_file():
            return None
        with np.load(p) as z:
            if bounds is not None and not np.array_equal(z["bounds"], np.asarray(bounds, dtype=np.int32)):
                raise ValueError(f"cached unit boundaries of {key} differ from the ones computed now")
            return z["emb"].astype(np.float32)

    def save(self, key: str, bounds: np.ndarray, emb: np.ndarray, record: dict | None = None) -> None:
        buf = io.BytesIO()
        np.savez(buf, bounds=np.asarray(bounds, dtype=np.int32), emb=np.asarray(emb, dtype=np.float16))
        atomic_write_bytes(self.path(key), buf.getvalue())
        if record is not None:
            self.log.append({"doc": key, **record})


def ingest_document(units: Units, context: str, embedder, cache: IndexCache, record: dict | None = None) -> tuple[np.ndarray, bool]:
    """Embeddings of the document's units, from the cache or computed and stored.
    Returns (embeddings float32 [n_units, dim], True if they were computed now)."""
    key = doc_hash(units.ids)
    emb = cache.load(key, units.bounds)
    if emb is not None:
        return emb, False
    import torch

    t0 = time.perf_counter()
    emb = embedder.encode_documents(units.texts(context))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    cache.save(key, units.bounds, emb, {
        "tokens": units.n_tokens, "units": units.n_units, "embed_s": round(seconds, 4),
        "tokens_per_s": round(units.n_tokens / seconds, 1) if seconds > 0 else None, **(record or {}),
    })
    # What a later load returns: the float16 values, so a cached and a fresh run rank identically.
    return np.asarray(emb, dtype=np.float16).astype(np.float32), True
