"""Synthetic generators: determinism, evidence offsets, lengths, deciles, needles (CPU only)."""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from pcm.tasks.common import get_tokenizer, load_task_config, occurrences
from pcm.tasks.synth import SYNTH_TASKS, BookPool, appears_in, ensure_book_cache, generate_example

TARGET = 2048


@pytest.fixture(scope="module")
def cfg():
    return load_task_config()


@pytest.fixture(scope="module")
def tok(cfg):
    return get_tokenizer(cfg["tokenizer"])


@pytest.fixture(scope="module")
def pool(tok, cfg):
    return BookPool(ensure_book_cache(tok, cfg, log=lambda m: None))


@pytest.fixture(scope="module")
def examples(tok, pool, cfg):
    """Two examples per task (indices 0 and 7: deciles 0 and 7)."""
    return {(t, i): generate_example(t, TARGET, 0, i, tok, pool, cfg=cfg) for t in SYNTH_TASKS for i in (0, 7)}


def test_vocabularies_disjoint(cfg):
    v = cfg["synth"]["vocab"]
    for key in ("key_adjectives", "key_nouns", "marker_words"):
        assert not set(v["test"][key]) & set(v["train"][key]), key
        assert len(set(v["test"][key])) == len(v["test"][key]), f"duplicate in test {key}"
    assert not set(v["test"]["name_first_letters"]) & set(v["train"]["name_first_letters"])
    (a, b), (c, d) = v["test"]["value_range"], v["train"]["value_range"]
    assert b < c or d < a


def test_deterministic_per_seed(tok, pool, cfg, examples):
    fresh = BookPool(pool.root)  # a new pool object: no state carried over
    for task in SYNTH_TASKS:
        again = generate_example(task, TARGET, 0, 7, tok, fresh, cfg=cfg)
        assert json.dumps(again, sort_keys=True) == json.dumps(examples[(task, 7)], sort_keys=True), task
        other = generate_example(task, TARGET, 1, 7, tok, fresh, cfg=cfg)
        assert other["context"] != again["context"], task


def test_book_mix_differs_per_seed(pool):
    a, b = pool.order("test", 0), pool.order("test", 1)
    assert sorted(a) == sorted(b) == list(range(len(pool)))
    assert not np.array_equal(a, b)
    assert np.array_equal(a, pool.order("test", 0))


def test_evidence_offsets_point_at_text(tok, examples):
    for (task, i), ex in examples.items():
        ids = tok(ex["context"], add_special_tokens=False)["input_ids"]
        for e in ex["evidence"] + ex["distractors"]:
            assert ex["context"][e["char_start"]:e["char_end"]] == e["text"], (task, i)
            decoded = tok.decode(ids[e["tok_start"]:e["tok_end"]])
            assert e["text"] in decoded, (task, i, decoded)
            assert len(decoded) <= len(e["text"]) + 8, (task, i, decoded)  # at most a merged space or newline around it


def test_lengths_within_tolerance(tok, cfg, examples):
    tol = cfg["synth"]["length_tolerance"]
    for (task, i), ex in examples.items():
        n = len(tok(ex["context"], add_special_tokens=False)["input_ids"])
        assert n == ex["length_tokens"], (task, i)
        assert math.ceil((1 - tol) * TARGET) <= n <= TARGET, (task, i, n)


def test_depth_deciles(examples):
    for (task, i), ex in examples.items():
        if task.startswith("S4"):
            assert ex["depth_decile"] is None
            continue
        assert ex["depth_decile"] == i % 10
        designated = [e for e in ex["evidence"] if e["designated"]]
        assert len(designated) == 1 and designated[0]["decile"] == i % 10, (task, i)
        if task.startswith("S3"):
            deciles = [e["decile"] for e in ex["evidence"]]
            assert len(set(deciles)) == len(deciles), "gold chain statements must lie in different deciles"
            assert [e["hop"] for e in ex["evidence"]] == list(range(ex["params"]["hops"]))


def test_needles_absent_from_haystack_and_unambiguous(examples):
    for (task, i), ex in examples.items():
        ctx = ex["context"]
        # Remove every inserted sentence: what is left is the haystack.
        hay = ctx
        for e in sorted(ex["evidence"] + ex["distractors"], key=lambda e: -e["char_start"]):
            hay = hay[: e["char_start"]] + hay[e["char_end"]:]
        strings = [] if task == "S4_count" else list(ex["answers"]) + list(ex["exclude"])  # an S4_count answer is a count, not a needle
        p = ex["params"]
        strings += [p[k] for k in ("key",) if k in p] + p.get("distractor_keys", []) + p.get("chain", [])
        for c in p.get("distractor_chains", []):
            strings += c
        if task == "S4_freq":
            strings += list(p["counts"])
        for s in strings:
            assert not appears_in(hay, s), (task, i, s)
        if task in ("S1", "S2_multikey"):
            assert occurrences(ctx, p["key"]) == 1
        if task == "S2_multivalue":
            assert occurrences(ctx, p["key"]) == 4
        for v in ex["answers"] + ex["exclude"]:
            if v.isdigit() and not task.startswith("S4"):
                assert occurrences(ctx, v) == 1, (task, v)


def test_task_specific_answers(examples):
    for i in (0, 7):
        c = examples[("S4_count", i)]
        assert c["answers"] == [str(len(c["evidence"]))]
        f = examples[("S4_freq", i)]
        counts = f["params"]["counts"]
        top = max(counts.values())
        assert [w for w, k in counts.items() if k == top] == f["answers"]
        assert sum(counts.values()) == len(f["evidence"])
        for h in (2, 3, 4):
            ex = examples[(f"S3_h{h}", i)]
            chain = ex["params"]["chain"]
            assert re.search(rf"VAR {chain[0]}\b", ex["question"])
            assert ex["evidence"][-1]["text"].endswith(f"= {ex['answers'][0]}.")
            assert len(ex["distractors"]) == 3 * h
        mk = examples[("S2_multikey", i)]
        gold_adj, gold_noun = mk["params"]["key"].split("-")
        for k in mk["params"]["distractor_keys"]:
            a, n = k.split("-")
            assert (a == gold_adj) != (n == gold_noun), "a distractor key shares exactly one word with the gold key"


def test_train_split_disjoint(tok, pool, cfg):
    with pytest.raises(ValueError):
        generate_example("S1", TARGET, 5, 0, tok, pool, split="train", cfg=cfg)
    with pytest.raises(ValueError):
        generate_example("S1", TARGET, cfg["synth"]["seed_limit"], 0, tok, pool, split="test", cfg=cfg)
    ex = generate_example("S1", TARGET, cfg["synth"]["seed_limit"], 0, tok, pool, split="train", cfg=cfg)
    v = cfg["synth"]["vocab"]["train"]
    adj, noun = ex["params"]["key"].split("-")
    assert adj in v["key_adjectives"] and noun in v["key_nouns"]
    lo, hi = v["value_range"]
    assert lo <= int(ex["answers"][0]) <= hi
    assert ex["id"].startswith("train-")
