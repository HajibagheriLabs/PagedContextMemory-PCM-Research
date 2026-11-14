"""Synthetic long-context tasks S1 to S4 over a PG-19 haystack (specification, section 5.1).

S1 single needle; S2 multi-key (four similar keys, ask one) and multi-value (one
key, four values, ask all); S3 chain of variable bindings (2, 3 or 4 hops, plus
distractor chains); S4 aggregation (count one marker, or name the most frequent
of several markers).

Haystack: PG-19 validation and test books. Each book is tokenized once and cached
on disk with the character offset of every token (`ensure_book_cache`), so a
haystack of a given token count is cut without re-tokenizing books. The book order
is a permutation fixed by (split, seed): every seed sees a different book mix.

Every example is a pure function of (split, task, target length, seed, index):
three random streams are derived from those values (haystack, content, positions),
so regenerating an example reproduces it exactly. The example's evidence depth
decile is index mod 10. Inserted sentences go at sentence boundaries. The final
context is tokenized once more: its measured length must lie within the tolerance
below the target, the designated evidence must lie in the requested decile, and
every needle string must occur exactly as often as intended (needle strings are
checked absent from the haystack before insertion). Otherwise the example is
rebuilt deterministically.

Train split: disjoint seeds (>= seed_limit), key words, value range, variable
names and marker words. Same PG-19 pool (the PG-19 train split is not on this
machine; see DEVIATIONS.md).
"""
from __future__ import annotations

import io
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pcm.runlog import atomic_write_bytes, atomic_write_json, atomic_write_text
from pcm.tasks.common import (
    SENTENCE_END,
    char_span_to_tokens,
    count_tokens,
    depth_decile,
    load_task_config,
    occurrences,
    repo_path,
    stable_int,
    tokenize_with_offsets,
    tokenizer_tag,
)

SYNTH_TASKS = ("S1", "S2_multikey", "S2_multivalue", "S3_h2", "S3_h3", "S3_h4", "S4_count", "S4_freq")
SPLITS = ("test", "train")


class GenerationError(RuntimeError):
    """An example could not be built within the attempt budget."""


# ---------------------------------------------------------------- book cache
def ensure_book_cache(tokenizer, cfg: dict | None = None, log=print) -> Path:
    """Tokenize every haystack book once and store text + token offsets. Resumable.

    Layout: <cache_dir>/<tokenizer tag>/<split>_<idx>.txt and .npy (int32 start
    character of each token, plus len(text) at the end), and index.json, which
    lists only books whose two files are complete. Rewritten atomically after each book.
    """
    from pcm.tasks.pg19 import stream_books

    cfg = cfg or load_task_config()
    root = repo_path(cfg["synth"]["cache_dir"]) / tokenizer_tag(tokenizer)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"books": []}
    have = {b["key"] for b in index["books"]}
    for split in cfg["synth"]["haystack_splits"]:
        for idx, book in enumerate(stream_books(split, columns=("short_book_title", "text"))):
            key = f"{split}_{idx:04d}"
            if key in have:
                continue
            text = book["text"]
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            starts = np.fromiter((a for a, _ in enc["offset_mapping"]), dtype=np.int64, count=len(enc["input_ids"]))
            offsets = np.append(starts, len(text)).astype(np.int32)
            buf = io.BytesIO()
            np.save(buf, offsets)
            atomic_write_text(root / f"{key}.txt", text)
            atomic_write_bytes(root / f"{key}.npy", buf.getvalue())
            index["books"].append(
                {"key": key, "split": split, "idx": idx, "title": book["short_book_title"], "chars": len(text), "tokens": int(len(offsets) - 1)}
            )
            atomic_write_json(index_path, index)
            log(f"book cache: {key} {len(offsets) - 1} tokens")
    index["tokenizer"] = tokenizer_tag(tokenizer)
    atomic_write_json(index_path, index)
    return root


class BookPool:
    """Read access to the cached books, with a small in-memory LRU of book texts."""

    def __init__(self, root: Path, max_cached: int = 12):
        self.root = Path(root)
        index = json.loads((self.root / "index.json").read_text(encoding="utf-8"))
        self.books = sorted(index["books"], key=lambda b: b["key"])
        if not self.books:
            raise FileNotFoundError(f"empty book cache at {root}")
        self._cache: OrderedDict[str, tuple[str, np.ndarray]] = OrderedDict()
        self.max_cached = max_cached

    def __len__(self) -> int:
        return len(self.books)

    def get(self, i: int) -> tuple[str, np.ndarray]:
        key = self.books[i]["key"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        text = (self.root / f"{key}.txt").read_text(encoding="utf-8")
        offsets = np.load(self.root / f"{key}.npy")
        self._cache[key] = (text, offsets)
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return text, offsets

    def order(self, split: str, seed: int) -> np.ndarray:
        """The book order for (split, seed): a permutation of the pool."""
        return np.random.default_rng(stable_int("book-order", split, seed)).permutation(len(self.books))


@dataclass
class Haystack:
    text: str
    offsets: np.ndarray  # approximate start character of each haystack token
    pieces: list[dict]


def build_haystack(pool: BookPool, order: np.ndarray, start_pos: int, start_frac: float, n_tokens: int, sep: str) -> Haystack:
    """About n_tokens of book text: from a paragraph start in book order[start_pos], then the following books."""
    parts: list[str] = []
    offs: list[np.ndarray] = []
    pieces: list[dict] = []
    cursor = 0
    remaining = n_tokens
    pos = start_pos
    first = True
    while remaining > 0:
        bi = int(order[pos % len(order)])
        text, off = pool.get(bi)
        n_book = len(off) - 1
        if first:
            j = int(start_frac * max(0, n_book - 1))
            c = int(off[j])
            m = re.compile(r"\n\s*\n\s*").search(text, c)
            c = m.end() if m and m.end() < len(text) else c
        else:
            m = re.compile(r"\S").search(text)
            c = m.start() if m else 0
        j0 = int(np.searchsorted(off[:-1], c, side="left"))
        j1 = min(n_book, j0 + remaining)
        if j1 <= j0:
            pos += 1
            first = False
            continue
        c0, c1 = int(off[j0]), int(off[j1])
        if parts:
            parts.append(sep)
            cursor += len(sep)
        seg = text[c0:c1]
        parts.append(seg)
        offs.append(off[j0:j1].astype(np.int64) - c0 + cursor)
        b = pool.books[bi]
        pieces.append({"book": b["key"], "title": b["title"], "char_start": c0, "char_end": c1})
        cursor += len(seg)
        remaining -= (j1 - j0) + 1
        pos += 1
        first = False
        if pos - start_pos > len(order):
            raise GenerationError("book pool exhausted")
    return Haystack(text="".join(parts), offsets=np.concatenate(offs), pieces=pieces)


# ---------------------------------------------------------------- content
@dataclass
class Item:
    """One inserted sentence."""

    text: str
    role: str                 # "evidence" or "distractor"
    hop: int = 0
    lo: float = 0.0           # allowed depth range (fraction of the context)
    hi: float = 1.0
    target: float | None = None
    designated: bool = False  # its decile is the example's stratified decile


@dataclass
class Content:
    items: list[Item]
    question: str
    answers: list[str]
    exclude: list[str]
    check_absent: list[str]           # strings that must not occur in the haystack
    expected_counts: dict[str, int]   # string -> exact number of occurrences in the context
    params: dict = field(default_factory=dict)


def _keys(vocab: dict, rng: np.random.Generator) -> tuple[str, str]:
    return str(rng.choice(vocab["key_adjectives"])), str(rng.choice(vocab["key_nouns"]))


def _values(vocab: dict, rng: np.random.Generator, k: int) -> list[str]:
    lo, hi = vocab["value_range"]
    out: list[str] = []  # a list, not a set: set order of strings changes between processes
    while len(out) < k:
        v = str(int(rng.integers(lo, hi + 1)))
        if v not in out:
            out.append(v)
    return out


def _names(vocab: dict, rng: np.random.Generator, k: int) -> list[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    firsts = vocab["name_first_letters"]
    out: list[str] = []
    while len(out) < k:
        name = str(rng.choice(list(firsts))) + "".join(rng.choice(list(letters), 4))
        if name not in out:
            out.append(name)
    return out


def _count_for(rng, per_8k: list, target_tokens: int, minimum: int) -> int:
    lo, hi = per_8k
    return max(minimum, int(round(rng.uniform(lo, hi) * target_tokens / 8192)))


def sample_content(task: str, tcfg: dict, vocab: dict, rng: np.random.Generator, target_tokens: int) -> Content:
    if task == "S1":
        adj, noun = _keys(vocab, rng)
        key = f"{adj}-{noun}"
        (value,) = _values(vocab, rng, 1)
        items = [Item(tcfg["needle"].format(key=key, value=value), "evidence", 0, designated=True)]
        return Content(items, tcfg["question"].format(key=key), [value], [], [key, value], {key: 1, value: 1}, {"key": key})

    if task == "S2_multikey":
        adj, noun = _keys(vocab, rng)
        gold = f"{adj}-{noun}"
        similar = [f"{adj}-{n}" for n in vocab["key_nouns"] if n != noun] + [f"{a}-{noun}" for a in vocab["key_adjectives"] if a != adj]
        k = tcfg["n_needles"] - 1
        others = [similar[i] for i in rng.choice(len(similar), size=k, replace=False)]
        values = _values(vocab, rng, k + 1)
        items = [Item(tcfg["needle"].format(key=gold, value=values[0]), "evidence", 0, designated=True)]
        items += [Item(tcfg["needle"].format(key=o, value=v), "distractor") for o, v in zip(others, values[1:])]
        counts = {s: 1 for s in [gold, *others, *values]}
        return Content(items, tcfg["question"].format(key=gold), [values[0]], values[1:], [gold, *others, *values], counts, {"key": gold, "distractor_keys": others})

    if task == "S2_multivalue":
        adj, noun = _keys(vocab, rng)
        key = f"{adj}-{noun}"
        values = _values(vocab, rng, tcfg["n_values"])
        items = [Item(tcfg["needle"].format(key=key, value=v), "evidence", 0, designated=(i == 0)) for i, v in enumerate(values)]
        counts = {key: len(values), **{v: 1 for v in values}}
        return Content(items, tcfg["question"].format(key=key), values, [], [key, *values], counts, {"key": key})

    if task.startswith("S3"):
        h = tcfg["hops"]
        n_chains = 1 + tcfg["distractor_chains"]
        names = _names(vocab, rng, h * n_chains)
        values = _values(vocab, rng, n_chains)
        items: list[Item] = []
        counts: dict[str, int] = {}
        chains = []
        for c in range(n_chains):
            ns = names[c * h : (c + 1) * h]  # ns[0] is asked about; ns[-1] holds the value
            chains.append(ns)
            role = "evidence" if c == 0 else "distractor"
            for k in range(h):
                if k < h - 1:
                    text = tcfg["link_statement"].format(name=ns[k], prev=ns[k + 1])
                else:
                    text = tcfg["value_statement"].format(name=ns[k], value=values[c])
                items.append(Item(text, role, k, designated=(c == 0 and k == h - 1)))
            for k, n in enumerate(ns):
                counts[n] = 1 if k == 0 else 2
            counts[values[c]] = 1
        return Content(
            items, tcfg["question"].format(name=chains[0][0]), [values[0]], values[1:], [*names, *values], counts,
            {"hops": h, "chain": chains[0], "distractor_chains": chains[1:]},
        )

    if task == "S4_count":
        adj, noun = _keys(vocab, rng)
        key = f"{adj}-{noun}"
        m = _count_for(rng, tcfg["markers_per_8k"], target_tokens, 2)
        items = [Item(tcfg["marker"].format(key=key), "evidence", 0) for _ in range(m)]
        return Content(items, tcfg["question"].format(key=key), [str(m)], [], [key], {key: m}, {"key": key, "count": m})

    if task == "S4_freq":
        k = tcfg["n_candidates"]
        words = [str(w) for w in rng.choice(vocab["marker_words"], size=k, replace=False)]
        m = _count_for(rng, tcfg["markers_per_8k"], target_tokens, k + 2)
        w = np.asarray(tcfg["weights"][:k], dtype=float)
        counts_arr = 1 + rng.multinomial(m - k, w / w.sum())
        while True:  # make the maximum unique: move one marker from a tied candidate to the first one
            tops = np.flatnonzero(counts_arr == counts_arr.max())
            if len(tops) == 1:
                break
            counts_arr[tops[0]] += 1
            counts_arr[tops[1]] -= 1  # the maximum is >= 2 because m >= k + 2
        top = int(np.argmax(counts_arr))
        counts = {words[i]: int(counts_arr[i]) for i in range(k)}
        items = [Item(tcfg["marker"].format(word=words[i]), "evidence", 0) for i in range(k) for _ in range(int(counts_arr[i]))]
        return Content(items, tcfg["question"], [words[top]], [x for i, x in enumerate(words) if i != top], words, counts, {"counts": counts})

    raise ValueError(f"unknown synthetic task {task!r}")


def place_items(task: str, content: Content, rng: np.random.Generator, decile: int | None, margin: float) -> None:
    """Fix every item's target depth. Designated item: inside `decile`. S3 gold hops: distinct other deciles."""
    def inside(d: int) -> float:
        return float(rng.uniform(d / 10 + margin, (d + 1) / 10 - margin))

    gold_deciles: list[int] = []
    if task.startswith("S3") and decile is not None:
        others = [d for d in range(10) if d != decile]
        n_other = sum(1 for it in content.items if it.role == "evidence" and not it.designated)
        gold_deciles = [int(d) for d in rng.choice(others, size=n_other, replace=False)]
    gi = 0
    for it in content.items:
        if it.designated and decile is not None:
            it.lo, it.hi = decile / 10, (decile + 1) / 10
            it.target = inside(decile)
        elif task.startswith("S3") and it.role == "evidence" and decile is not None:
            d = gold_deciles[gi]
            gi += 1
            it.lo, it.hi = d / 10, (d + 1) / 10
            it.target = inside(d)
        else:
            it.lo, it.hi = 0.0, 1.0
            it.target = float(rng.uniform(0.0, 1.0))


def _snap(text: str, c: int, lo_c: int, hi_c: int) -> int:
    """Insertion point at a sentence end in [lo_c, hi_c], nearest to c; else at whitespace; else c."""
    fwd = SENTENCE_END.search(text, c, min(len(text), hi_c + 1))
    back = None
    for m in SENTENCE_END.finditer(text, max(lo_c, c - 4000), c):
        back = m
    cands = [m.end() for m in (fwd, back) if m is not None and lo_c <= m.end() <= hi_c]
    if cands:
        return min(cands, key=lambda p: abs(p - c))
    ws = [m.start() for m in re.finditer(r"\s", text[lo_c : hi_c + 1])]
    if ws:
        return lo_c + min(ws, key=lambda p: abs(lo_c + p - c))
    return c


def insert_items(hay: Haystack, items: list[Item]) -> tuple[str, list[tuple[int, int]]]:
    """Insert ' ' + item.text at sentence ends near each target depth. Returns (context, char spans)."""
    n = len(hay.offsets)
    text = hay.text

    def char_at(frac: float) -> int:
        return int(hay.offsets[min(n - 1, max(0, int(frac * n)))]) if frac < 1.0 else len(text)

    points = []
    for i, it in enumerate(items):
        p = _snap(text, char_at(it.target), char_at(it.lo), char_at(it.hi))
        points.append((p, i))
    points.sort()
    out: list[str] = []
    spans: list[tuple[int, int]] = [(0, 0)] * len(items)
    prev = 0
    shift = 0
    for p, i in points:
        out.append(text[prev:p])
        ins = " " + items[i].text
        out.append(ins)
        start = p + shift + 1
        spans[i] = (start, start + len(items[i].text))
        shift += len(ins)
        prev = p
    out.append(text[prev:])
    return "".join(out), spans


# ---------------------------------------------------------------- examples
def example_id(split: str, task: str, length_name: str, seed: int, index: int) -> str:
    return f"{'' if split == 'test' else 'train-'}{task}-{length_name}-s{seed}-{index:04d}"


def _check_seed(split: str, seed: int, limit: int) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "test" and not 0 <= seed < limit:
        raise ValueError(f"test seeds are in [0, {limit})")
    if split == "train" and seed < limit:
        raise ValueError(f"train seeds are >= {limit}, disjoint from test seeds")


def appears_in(text: str, s: str) -> bool:
    """True if s occurs in text: digits not touching other digits; words at a word start, any case and suffix."""
    if s.isdigit():
        return re.search(rf"(?<!\d){re.escape(s)}(?!\d)", text) is not None
    return re.search(rf"(?<![\w-]){re.escape(s)}", text, re.IGNORECASE) is not None


def generate_example(
    task: str,
    target_tokens: int,
    seed: int,
    index: int,
    tokenizer,
    pool: BookPool,
    split: str = "test",
    length_name: str | None = None,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or load_task_config()
    scfg = cfg["synth"]
    _check_seed(split, seed, scfg["seed_limit"])
    tcfg = scfg["tasks"][task]
    vocab = scfg["vocab"][split]
    length_name = length_name or str(target_tokens)
    tol = scfg["length_tolerance"]
    lo_len = math.ceil((1 - tol) * target_tokens)
    hi_len = target_tokens
    aim = target_tokens - max(2, int(0.002 * target_tokens))
    window, reserve = scfg["native_window"], scfg["prompt_reserve"]
    if target_tokens <= window < target_tokens + reserve:
        # The cell is meant to fit the native window with its prompt: leave room for wrapper and answer.
        hi_len = window - reserve
        aim = min(aim, hi_len - max(2, int(0.002 * target_tokens)))
        if hi_len < lo_len:
            raise ValueError(f"prompt_reserve {reserve} leaves no room inside the {tol:.0%} tolerance of {target_tokens}")
    decile = None if task.startswith("S4") else index % 10

    base = (split, task, target_tokens, seed, index)
    rng_hay = np.random.default_rng(stable_int(*base, "haystack"))
    rng_content = np.random.default_rng(stable_int(*base, "content"))
    order = pool.order(split, seed)
    start_pos = int(rng_hay.integers(len(order)))
    start_frac = float(rng_hay.uniform(0.0, 0.9))

    content = sample_content(task, tcfg, vocab, rng_content, target_tokens)
    item_tokens = sum(count_tokens(tokenizer, " " + it.text) for it in content.items)
    hay_tokens = max(16, aim - item_tokens)
    last_problem = None
    for attempt in range(scfg["max_attempts"] * 2):
        hay = build_haystack(pool, order, start_pos, start_frac, hay_tokens, scfg["book_separator"])
        present = [s for s in content.check_absent if appears_in(hay.text, s)]
        if present:
            last_problem = f"needle strings occur in the haystack: {present}"
            content = sample_content(task, tcfg, vocab, rng_content, target_tokens)
            item_tokens = sum(count_tokens(tokenizer, " " + it.text) for it in content.items)
            hay_tokens = max(16, aim - item_tokens)
            continue
        rng_pos = np.random.default_rng(stable_int(*base, "positions", attempt))
        place_items(task, content, rng_pos, decile, scfg["depth_margin"])
        context, spans = insert_items(hay, content.items)
        ids, starts, ends = tokenize_with_offsets(tokenizer, context)
        n = len(ids)
        if not lo_len <= n <= hi_len:
            last_problem = f"length {n} outside [{lo_len}, {hi_len}]"
            hay_tokens = max(16, hay_tokens + (aim - n))
            continue
        records = []
        for it, (c0, c1) in zip(content.items, spans):
            t0, t1 = char_span_to_tokens(starts, ends, c0, c1)
            records.append(
                {"text": it.text, "role": it.role, "hop": it.hop, "char_start": c0, "char_end": c1,
                 "tok_start": t0, "tok_end": t1, "depth": round(t0 / n, 6), "decile": depth_decile(t0, n), "designated": it.designated}
            )
        problems = []
        for it, r in zip(content.items, records):
            if it.lo > 0.0 or it.hi < 1.0:
                if r["decile"] != int(round(it.lo * 10)):
                    problems.append(f"item at decile {r['decile']}, wanted {int(round(it.lo * 10))}")
        for s, k in content.expected_counts.items():
            got = occurrences(context, s, ignore_case=not s.isupper())
            if got != k:
                problems.append(f"{s!r} occurs {got} times, expected {k}")
        if problems:
            last_problem = "; ".join(problems)
            continue
        evidence = [r for r in records if r["role"] == "evidence"]
        distractors = [r for r in records if r["role"] == "distractor"]
        evidence.sort(key=lambda r: (r["hop"], r["char_start"]))
        distractors.sort(key=lambda r: r["char_start"])
        return {
            "id": example_id(split, task, length_name, seed, index),
            "task": task,
            "family": tcfg["family"],
            "split": split,
            "seed": seed,
            "index": index,
            "length_name": length_name,
            "length_nominal": target_tokens,
            "length_tokens": n,
            "tokenizer": tokenizer_tag(tokenizer),
            "instruction": tcfg["instruction"],
            "question": content.question,
            "answers": content.answers,
            "exclude": content.exclude,
            "scoring": tcfg["scoring"],
            "max_new_tokens": tcfg["max_new_tokens"],
            "chance": tcfg["chance"],
            "depth_decile": decile,
            "evidence": evidence,
            "distractors": distractors,
            "haystack": hay.pieces,
            "params": content.params,
            "attempts": attempt + 1,
            "context": context,
        }
    raise GenerationError(f"{example_id(split, task, length_name, seed, index)}: {last_problem}")


def prompt_parts(example: dict, cfg: dict | None = None) -> tuple[str, str, str]:
    """(system, user text before the context, user text after it) for a synthetic example."""
    p = (cfg or load_task_config())["synth"]["prompt"]
    return p["system"], p["before"].format(instruction=example["instruction"]), p["after"].format(question=example["question"])


def data_file(task: str, length_name: str, seed: int, cfg: dict | None = None, split: str = "test") -> Path:
    cfg = cfg or load_task_config()
    root = repo_path(cfg["synth"]["data_dir"])
    sub = task if split == "test" else f"train/{task}"
    return root / sub / f"{length_name}_s{seed}.jsonl"
