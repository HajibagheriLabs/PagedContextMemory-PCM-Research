"""Helpers shared by the task modules: configuration, tokenizer, offsets, depth.

Every length in this project is counted in tokens of the reader's tokenizer
(both readers share it). Offsets come in two kinds and both are recorded for
every piece of evidence: characters in the context string, and token indices in
the token ids of the context string tokenized on its own (no special tokens).
"""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_YAML = REPO_ROOT / "configs" / "a2_tasks.yaml"

# Where a sentence may end: terminal punctuation, optional closing quotes or brackets.
SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s)")


def load_task_config(path: Path | None = None) -> dict:
    with Path(path or TASKS_YAML).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=4)
def get_tokenizer(key: str = "reader"):
    """The fast tokenizer of a reader (key of configs/assets.yaml or repo id), loaded from the local cache."""
    from transformers import AutoTokenizer

    from pcm.hub import model_path

    tok = AutoTokenizer.from_pretrained(model_path(key))
    tok.model_max_length = 10**12  # lengths are measured here, never truncated; silences the length warning
    return tok


@lru_cache(maxsize=8)
def _tag_of(serialised: str) -> str:
    return "tok-" + hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:12]


def tokenizer_tag(tokenizer) -> str:
    """Stable identifier of a tokenizer: a hash of its full serialised definition.

    Two models with the same tokenizer (both Qwen3 readers) get the same tag.
    """
    tag = getattr(tokenizer, "_pcm_tag", None)
    if tag is None:
        tag = _tag_of(tokenizer.backend_tokenizer.to_str())
        try:
            tokenizer._pcm_tag = tag
        except AttributeError:
            pass
    return tag


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], list[int], list[int]]:
    """(ids, starts, ends): token ids and the character span of each token."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [a for a, _ in enc["offset_mapping"]]
    ends = [b for _, b in enc["offset_mapping"]]
    return enc["input_ids"], starts, ends


def char_span_to_tokens(starts: Sequence[int], ends: Sequence[int], char_start: int, char_end: int) -> tuple[int, int]:
    """Token range [t0, t1) covering the characters [char_start, char_end).

    t0 is the first token that ends after char_start, t1 one past the last token
    that starts before char_end. A token that straddles the boundary (a leading
    space merged into the first word, say) is included.
    """
    if char_end <= char_start:
        raise ValueError("empty character span")
    t0 = bisect_right(ends, char_start)
    t1 = bisect_left(starts, char_end)
    if t1 <= t0:
        raise ValueError(f"no token covers characters {char_start}..{char_end}")
    return t0, t1


def depth_decile(tok_start: int, n_tokens: int) -> int:
    """Decile 0..9 of a position: floor(10 * tok_start / n_tokens), clipped to 9."""
    return min(9, int(10 * tok_start / n_tokens))


def stable_int(*parts) -> int:
    """A 64-bit integer derived from the parts' text, the same on every machine and run."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def config_hash(obj) -> str:
    """Short hash of a JSON-serialisable object, independent of key order."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:12]


def iter_jsonl(path: Path):
    """Rows of a JSON-lines file, read one line at a time."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def word_pattern(word: str, ignore_case: bool = True) -> re.Pattern:
    """Whole-word pattern; digit strings must not touch other digits."""
    if word.isdigit():
        return re.compile(rf"(?<!\d){re.escape(word)}(?!\d)")
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])", flags)


def occurrences(text: str, word: str, ignore_case: bool = True) -> int:
    return len(word_pattern(word, ignore_case).findall(text))
