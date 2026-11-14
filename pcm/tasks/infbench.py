"""InfiniteBench: Retrieve.PassKey, Retrieve.Number, Retrieve.KV and En.MC (specification, section 5.1).

Data: one JSON-lines file per task in the local hub cache (fields id, context,
input, answer, options). Contexts reach 1.1M characters, so files are read one
line at a time.

Prompt: the official chat-model path of third_party/InfiniteBench/src/eval_utils.py,
`create_msgs` with the chat template set named in configs/a2_tasks.yaml
(infbench.template_set) and no tokenizer (the official script would otherwise cut
the input to 128K tokens in the middle; truncation is a method here, not part of
the data). Its system message and template are used as they are.
Label: the official `get_answer`. Scorer: the official `get_score_one_<task>` of
src/compute_scores.py. max_new_tokens: the official DATA_NAME_TO_MAX_NEW_TOKENS.

Evidence (not part of the benchmark, needed for oracle pages and evidence recall):
for passkey and number_string the needle sentence pair that states the answer, for
kv_retrieval the "key": "value" pair of the asked key, none for longbook_choice_eng.

Measured token lengths and evidence offsets are cached per task under
data/meta/infbench/<tokenizer tag>/<task>.jsonl, one line per sample.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

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
from pcm.tasks.official import infbench as official

TASKS = ("passkey", "number_string", "kv_retrieval", "longbook_choice_eng")
SCORER_MODEL_NAME = "pcm"  # the official scorers take a model name; only math_calc (unused) reads it


def data_file(task: str) -> Path:
    a = load_assets()["datasets"]["infbench"]
    root = dataset_path(a["repo"], a["files"])
    path = root / f"{task}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def iter_rows(task: str) -> Iterator[dict]:
    if task not in TASKS:
        raise ValueError(f"unsupported InfiniteBench task {task!r}")
    with data_file(task).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prompt_parts(row: dict, task: str, cfg: dict | None = None) -> tuple[str, str, str]:
    """(system, user text before the context, user text after it), official chat path."""
    template_set = (cfg or load_task_config())["infbench"]["template_set"]
    eg = {k: (list(v) if isinstance(v, list) else v) for k, v in row.items()}
    eg["context"] = CONTEXT_SENTINEL
    msgs, _ = official().create_msgs(None, eg, task, ".", model_name=template_set)
    if len(msgs) != 2 or msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
        raise ValueError(f"unexpected official message layout for {task}")
    user = msgs[1]["content"]
    if user.count(CONTEXT_SENTINEL) != 1:
        raise ValueError("official template did not keep the context placeholder exactly once")
    before, after = user.split(CONTEXT_SENTINEL)
    return msgs[0]["content"], before, after


def label(row: dict, task: str):
    return official().get_answer(row, task)


def score(prediction: str, example: dict) -> float:
    """Official InfiniteBench scorer for the example's task: 1.0 or 0.0."""
    return float(official().scorers[example["task"]](prediction, example["label"], SCORER_MODEL_NAME))


def evidence_spans(row: dict, task: str) -> list[tuple[int, int]]:
    ctx = row["context"]
    ans = row["answer"][0] if isinstance(row["answer"], list) else row["answer"]
    if task == "passkey":
        pat = rf"The pass key is {re.escape(str(ans))}\. Remember it\. {re.escape(str(ans))} is the pass key\."
    elif task == "number_string":
        pat = rf"The sequence of digits is {re.escape(str(ans))}\. Remember it\. {re.escape(str(ans))} is the sequence of digits\."
    elif task == "kv_retrieval":
        m = re.search(r'Key: "([^"]+)"', row["input"])
        if not m:
            return []
        pat = re.escape(f'"{m.group(1)}": "{ans}"')
    else:
        return []
    return [(m.start(), m.end()) for m in re.finditer(pat, ctx)]


def meta_file(task: str, tokenizer, cfg: dict | None = None) -> Path:
    cfg = cfg or load_task_config()
    return repo_path(cfg["infbench"]["meta_dir"]) / tokenizer_tag(tokenizer) / f"{task}.jsonl"


def _measure(row: dict, task: str, tokenizer, cfg: dict | None = None) -> dict:
    ctx = row["context"]
    ids, starts, ends = tokenize_with_offsets(tokenizer, ctx)
    n = len(ids)
    ev = []
    for a, b in evidence_spans(row, task):
        t0, t1 = char_span_to_tokens(starts, ends, a, b)
        ev.append({"text": ctx[a:b], "char_start": a, "char_end": b, "tok_start": t0, "tok_end": t1, "hop": 0,
                   "depth": round(t0 / n, 6), "decile": depth_decile(t0, n)})
    system, before, after = prompt_parts(row, task, cfg)
    chat = chat_segments(tokenizer, system, before, after, ctx)
    prompt_tokens = count_tokens(tokenizer, chat.prefix) + n + count_tokens(tokenizer, chat.suffix)
    return {"id": row["id"], "length_tokens": n, "prompt_tokens": prompt_tokens, "chars": len(ctx), "evidence": ev}


def ensure_meta(task: str, tokenizer, cfg: dict | None = None, log=None, limit: int | None = None) -> dict:
    """Measured lengths and evidence offsets of the rows of `task` (the first `limit`, or all),
    cached one line per row. Resumable."""
    jl = JsonlLog(meta_file(task, tokenizer, cfg), id_field="id")
    have = {r["id"]: r for r in jl.rows()}
    for k, row in enumerate(iter_rows(task)):
        if limit is not None and k >= limit:
            break
        if row["id"] in have:
            continue
        rec = _measure(row, task, tokenizer, cfg)
        jl.append(rec)
        have[row["id"]] = rec
        if log and k % 50 == 0:
            log(f"infbench {task}: {k} rows measured")
    return have


def load(task: str, tokenizer, cfg: dict | None = None, n: int | None = None) -> list[dict]:
    """Examples of one task in the harness's common form (the first n rows in file order)."""
    cfg = cfg or load_task_config()
    meta = ensure_meta(task, tokenizer, cfg, limit=n)
    off = official()
    out = []
    for k, row in enumerate(iter_rows(task)):
        if n is not None and k >= n:
            break
        m = meta[row["id"]]
        out.append({
            "id": f"infbench-{task}-{row['id']}",
            "benchmark": "infbench",
            "family": "infbench",
            "task": task,
            "length_name": "full",
            "index": k,
            "row": {"id": row["id"], "input": row["input"], "answer": row["answer"], "options": row["options"]},
            "context": row["context"],
            "question": row["input"],
            "label": label(row, task),
            "answers": row["answer"],
            "length_tokens": m["length_tokens"],
            "prompt_tokens": m["prompt_tokens"],
            "evidence": m["evidence"],
            "depth_decile": m["evidence"][0]["decile"] if m["evidence"] else None,
            "max_new_tokens": off.max_new_tokens[task],
            "chance": cfg["infbench"]["chance"][task],
        })
    return out
