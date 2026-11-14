"""BABILong: official test set, official prompt, official scorer (specification, section 5.1).

Data: `RMT-team/babilong` in the local hub cache, file data/<task>/<length>.json,
100 rows per file with fields input, question, target. Row i of every length is
the same bAbI sample (same question and target) hidden in a different amount of
PG-19 text, so row i of the 0k split holds exactly the facts of row i at any length.

Prompt: `get_formatted_input` and `DEFAULT_PROMPTS` from third_party/babilong
(instruction, few-shot examples and post prompt), inside the chat template with
the system prompt of the official run scripts. The context is placed with the
official `.strip()`; character offsets refer to the stripped context.

Scorer: `compare_answers(target, output, question, TASK_LABELS[task])` from
third_party/babilong/babilong/metrics.py, the one the official result tables use.

Fact locator (section 5.1): a sentence of the context is a fact if every word in it
belongs to the vocabulary of the 0k split of the same task. Located facts carry
character and token offsets. Because rows align across lengths, the located facts
of row i can be checked against the sentences of row i at 0k (`locator_check`).

Measured lengths and located facts are cached per (task, length) under
data/meta/babilong/<tokenizer tag>/, one JSON line per sample, so a run never
tokenizes a 256K context twice.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from pcm.hub import dataset_path, load_assets
from pcm.models import CONTEXT_SENTINEL, chat_segments
from pcm.runlog import JsonlLog
from pcm.tasks.common import (
    char_span_to_tokens,
    count_tokens,
    depth_decile,
    load_task_config,
    repo_path,
    tokenize_with_offsets,
    tokenizer_tag,
)
from pcm.tasks.official import babilong as official

WORD = re.compile(r"[A-Za-z]+")
# A sentence: from a non-space character to the next terminal punctuation (with closing quotes).
SENTENCE = re.compile(r"[^\s.!?][^.!?]*[.!?]+[\"'”’)\]]*")


def data_file(task: str, length: str) -> Path:
    repo = load_assets()["datasets"]["babilong"]["repo"]
    root = dataset_path(repo, [f"data/{task}/{length}.json"])
    path = root / "data" / task / f"{length}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_rows(task: str, length: str) -> list[dict]:
    """The rows of one (task, length) file. One file is the streaming unit (100 rows)."""
    with data_file(task, length).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows or not {"input", "question", "target"} <= set(rows[0]):
        raise ValueError(f"unexpected BABILong file layout for {task}/{length}")
    return rows


@lru_cache(maxsize=32)
def vocabulary(task: str) -> frozenset[str]:
    """Every word (case kept) in the inputs of the 0k split of `task`."""
    words: set[str] = set()
    for row in load_rows(task, "0k"):
        words.update(WORD.findall(row["input"]))
    return frozenset(words)


@lru_cache(maxsize=32)
def fact_set(task: str) -> frozenset[str]:
    """Every sentence of the 0k split of `task`: the set of bAbI facts a located sentence should belong to."""
    out: set[str] = set()
    for row in load_rows(task, "0k"):
        t = row["input"].strip()
        out.update(t[a:b] for a, b in sentences(t))
    return frozenset(out)


def sentences(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in SENTENCE.finditer(text)]


def locate_facts(text: str, vocab: frozenset[str], min_words: int = 2) -> list[tuple[int, int]]:
    """Character spans of sentences whose words all belong to `vocab` (and number at least min_words)."""
    out = []
    for a, b in sentences(text):
        words = WORD.findall(text[a:b])
        if len(words) >= min_words and all(w in vocab for w in words):
            # Trim to the fact proper: first letter to last terminal punctuation mark. Facts can sit
            # against quotes or brackets of the surrounding text, which the sentence pattern takes along.
            seg = text[a:b]
            first = re.search(r"[A-Za-z]", seg).start()
            last = max(seg.rfind("."), seg.rfind("!"), seg.rfind("?"))
            a2, b2 = a + first, a + last + 1
            # Several terminal marks in a row ("garden..") belong to the surrounding text: keep one.
            while b2 - 1 > a2 and text[b2 - 2] in ".!?":
                b2 -= 1
            out.append((a2, b2))
    return out


def prompt_parts(task: str, question: str, cfg: dict | None = None) -> tuple[str, str, str]:
    """(system, user text before the context, user text after it), official template."""
    bcfg = (cfg or load_task_config())["babilong"]
    off = official()
    p = off.DEFAULT_PROMPTS[task]
    text = off.get_formatted_input(
        CONTEXT_SENTINEL,
        question,
        p["examples"] if bcfg["use_examples"] else "",
        p["instruction"] if bcfg["use_instruction"] else "",
        p["post_prompt"] if bcfg["use_post_prompt"] else "",
        template=off.DEFAULT_TEMPLATE,
    )
    if text.count(CONTEXT_SENTINEL) != 1:
        raise ValueError("official template did not keep the context placeholder exactly once")
    before, after = text.split(CONTEXT_SENTINEL)
    return bcfg["system"], before, after


def score(prediction: str, example: dict) -> float:
    """Official BABILong scorer: 1.0 or 0.0."""
    off = official()
    return float(off.compare_answers(example["target"], prediction, example["question"], off.TASK_LABELS[example["task"]]))


def meta_file(task: str, length: str, tokenizer, cfg: dict | None = None) -> Path:
    cfg = cfg or load_task_config()
    return repo_path(cfg["babilong"]["meta_dir"]) / tokenizer_tag(tokenizer) / f"{task}_{length}.jsonl"


def _measure(task: str, length: str, i: int, row: dict, tokenizer, cfg: dict, facts_0k: list[str] | None) -> dict:
    context = row["input"].strip()
    ids, starts, ends = tokenize_with_offsets(tokenizer, context)
    n = len(ids)
    facts = []
    for a, b in locate_facts(context, vocabulary(task), cfg["babilong"]["min_fact_words"]):
        t0, t1 = char_span_to_tokens(starts, ends, a, b)
        facts.append({"text": context[a:b], "char_start": a, "char_end": b, "tok_start": t0, "tok_end": t1,
                      "hop": None, "depth": round(t0 / n, 6) if n else 0.0, "decile": depth_decile(t0, n) if n else 0})
    system, before, after = prompt_parts(task, row["question"], cfg)
    chat = chat_segments(tokenizer, system, before, after, context)
    prompt_tokens = count_tokens(tokenizer, chat.prefix) + n + count_tokens(tokenizer, chat.suffix)
    rec = {"index": i, "length_tokens": n, "prompt_tokens": prompt_tokens, "chars": len(context), "facts": facts}
    if facts_0k is not None:
        located = Counter(f["text"] for f in facts)
        gold = Counter(facts_0k)
        matched = sum((located & gold).values())
        rec["locator"] = {"facts_0k": sum(gold.values()), "located": sum(located.values()), "matched": matched}
    rec.setdefault("locator", {"located": len(facts)})
    known = fact_set(task)
    rec["locator"]["located_known"] = sum(1 for f in facts if f["text"] in known)  # located sentences that are 0k facts
    return rec


def ensure_meta(task: str, length: str, tokenizer, cfg: dict | None = None, rows: list[dict] | None = None) -> dict[int, dict]:
    """Measured lengths and located facts of every row, computed once and cached (one line per row)."""
    cfg = cfg or load_task_config()
    log = JsonlLog(meta_file(task, length, tokenizer, cfg), id_field="index")
    have = {r["index"]: r for r in log.rows()}
    rows = rows if rows is not None else load_rows(task, length)
    if len(have) < len(rows):
        rows0 = load_rows(task, "0k")
        for i, row in enumerate(rows):
            if i in have:
                continue
            same = i < len(rows0) and rows0[i]["question"] == row["question"] and rows0[i]["target"] == row["target"]
            facts_0k = None
            if same:  # row i at 0k is the same bAbI sample: its sentences are exactly the facts
                text0 = rows0[i]["input"].strip()
                facts_0k = [text0[a:b] for a, b in sentences(text0)]
            rec = _measure(task, length, i, row, tokenizer, cfg, facts_0k)
            log.append(rec)
            have[i] = rec
    return have


def load(task: str, length: str, tokenizer, cfg: dict | None = None, n: int | None = None) -> list[dict]:
    """Examples of one (task, length) cell in the harness's common form."""
    cfg = cfg or load_task_config()
    rows = load_rows(task, length)
    meta = ensure_meta(task, length, tokenizer, cfg, rows)
    out = []
    for i, row in enumerate(rows[: n if n is not None else len(rows)]):
        m = meta[i]
        out.append({
            "id": f"babilong-{task}-{length}-{i:03d}",
            "benchmark": "babilong",
            "family": "babilong",
            "task": task,
            "length_name": length,
            "index": i,
            "context": row["input"].strip(),
            "question": row["question"],
            "target": row["target"],
            "answers": [row["target"]],
            "length_tokens": m["length_tokens"],
            "prompt_tokens": m["prompt_tokens"],
            "evidence": m["facts"],
            "depth_decile": None,
            "max_new_tokens": cfg["babilong"]["max_new_tokens"],
            "chance": cfg["babilong"]["chance"][task],
        })
    return out
