"""Crash-safe result files.

The GPU on the local machine can drop off without warning, and the process or
the whole machine may go with it. Everything a run produces therefore goes
through two primitives:

- JsonlLog: an append-only file, one JSON object per line, forced to disk after
  every line. A rerun reads the ids already present and skips them. A line cut
  short by a crash is ignored on reading and never extended on writing.
- atomic_write_*: a file is written under a temporary name and renamed into
  place, so a reader sees the old content or the new content, never half of it.

The unit of work between two durable writes should take minutes at most: one
example in an evaluation, one document in ingest, a fixed number of steps in
training. A rerun of the same command continues from the last durable write.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterator


def _fsync(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        _fsync(f)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False))


class JsonlLog:
    """Append-only JSON-lines file with durable appends and tolerant reads."""

    def __init__(self, path: Path, id_field: str = "id"):
        self.path = Path(path)
        self.id_field = id_field
        self.bad_lines = 0

    def rows(self) -> Iterator[dict]:
        """Every well-formed row, in file order. Malformed lines are counted in bad_lines."""
        self.bad_lines = 0
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    self.bad_lines += 1
                    continue
                if isinstance(row, dict):
                    yield row
                else:
                    self.bad_lines += 1

    def done_ids(self, where: Callable[[dict], bool] | None = None) -> set:
        """Ids of the rows present (optionally only those for which `where` is true)."""
        return {
            row[self.id_field]
            for row in self.rows()
            if self.id_field in row and (where is None or where(row))
        }

    def _ends_mid_line(self) -> bool:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return False
        with self.path.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"

    def append(self, row: dict) -> None:
        """Write one row and force it to disk before returning."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        if "\n" in line:
            raise ValueError("a row must serialise to one line")
        # A crash can leave a partial last line. Close it off so this row starts on its own line.
        prefix = "\n" if self._ends_mid_line() else ""
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(prefix + line + "\n")
            _fsync(f)
