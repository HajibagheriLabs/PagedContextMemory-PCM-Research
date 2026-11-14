"""Streaming access to PG-19 books (haystack filler).

Only the validation and test splits are ever read. Books are yielded one at a
time from the parquet file in small record batches, so the dataset is never
held in memory whole.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pcm.hub import dataset_path, load_assets

ALLOWED_SPLITS = ("validation", "test")


def split_files(split: str, allow_download: bool = False) -> list[Path]:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"PG-19 split {split!r} is not allowed; use one of {ALLOWED_SPLITS}")
    repo = load_assets()["datasets"]["pg19"]["repo"]
    patterns = [f"data/{s}-*.parquet" for s in ALLOWED_SPLITS]
    root = dataset_path(repo, patterns, allow_download=allow_download)
    files = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet file for PG-19 split {split!r} under {root}")
    return files


def stream_books(
    split: str,
    columns: tuple[str, ...] | None = None,
    batch_size: int = 1,
    allow_download: bool = False,
) -> Iterator[dict]:
    """Yield one book (a dict of column -> value) at a time."""
    import pyarrow.parquet as pq

    for path in split_files(split, allow_download=allow_download):
        pf = pq.ParquetFile(path)
        cols = list(columns) if columns else None
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                yield row


def split_info(split: str) -> dict:
    """Row count and column names from parquet metadata only (no data read)."""
    import pyarrow.parquet as pq

    rows = 0
    columns: list[str] = []
    for path in split_files(split):
        pf = pq.ParquetFile(path)
        rows += pf.metadata.num_rows
        columns = pf.schema_arrow.names
    return {"split": split, "books": rows, "columns": columns}
