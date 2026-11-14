"""BABILong and InfiniteBench loaders: official prompts and scorers reproduced, facts located (CPU only)."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from pcm.tasks import babilong as B
from pcm.tasks import infbench as I
from pcm.tasks.common import get_tokenizer, load_task_config
from pcm.tasks.official import babilong as off_b
from pcm.tasks.official import infbench as off_i


@pytest.fixture(scope="module")
def cfg():
    return load_task_config()


@pytest.fixture(scope="module")
def tok(cfg):
    return get_tokenizer(cfg["tokenizer"])


# ---------------------------------------------------------------- BABILong
def test_babilong_prompt_is_the_official_one(cfg):
    row = B.load_rows("qa2", "1k")[3]
    system, before, after = B.prompt_parts("qa2", row["question"], cfg)
    p = off_b().DEFAULT_PROMPTS["qa2"]
    official = off_b().get_formatted_input(row["input"], row["question"], p["examples"], p["instruction"], p["post_prompt"],
                                           template=off_b().DEFAULT_TEMPLATE)
    assert before + row["input"].strip() + after == official
    assert system == "You are a helpful assistant."


def test_babilong_scorer_hand_cases():
    ex = {"task": "qa1", "target": "bathroom", "question": "Where is Mary?"}
    assert B.score("The most recent location of Mary is bathroom.", ex) == 1.0
    assert B.score("The most recent location of Mary is kitchen.", ex) == 0.0
    assert B.score("bathroom or kitchen", ex) == 0.0  # more than one label
    ex2 = {"task": "qa8", "target": "apple,milk", "question": "What is Daniel carrying?"}
    assert B.score("apple,milk", ex2) == 1.0 and B.score("apple", ex2) == 0.0
    ex3 = {"task": "qa6", "target": "no", "question": "Is John in the garden?"}
    assert B.score("no", ex3) == 1.0 and B.score("yes", ex3) == 0.0


def test_babilong_rows_align_across_lengths():
    r0, r4 = B.load_rows("qa1", "0k"), B.load_rows("qa1", "4k")
    assert len(r0) == len(r4) == 100
    assert all(a["question"] == b["question"] and a["target"] == b["target"] for a, b in zip(r0, r4))


def test_fact_locator_finds_the_0k_facts():
    """On qa1 and qa4 at 2k every 0k fact of the row is located, and nothing else."""
    for task in ("qa1", "qa4"):
        voc = B.vocabulary(task)
        r0, r2 = B.load_rows(task, "0k"), B.load_rows(task, "2k")
        for i in range(20):
            ctx = r2[i]["input"].strip()
            got = Counter(ctx[a:b] for a, b in B.locate_facts(ctx, voc))
            t0 = r0[i]["input"].strip()
            assert got == Counter(t0[a:b] for a, b in B.sentences(t0)), (task, i)


def test_babilong_load(tok, cfg):
    ex = B.load("qa1", "1k", tok, cfg, n=5)
    assert len(ex) == 5
    for e in ex:
        ids = tok(e["context"], add_special_tokens=False)["input_ids"]
        assert len(ids) == e["length_tokens"]
        assert e["prompt_tokens"] > e["length_tokens"]
        assert e["evidence"], e["id"]
        for f in e["evidence"]:
            assert e["context"][f["char_start"]:f["char_end"]] == f["text"]
            assert f["text"] in tok.decode(ids[f["tok_start"]:f["tok_end"]])


# ---------------------------------------------------------------- InfiniteBench
@pytest.fixture(scope="module")
def first_rows():
    out = {}
    for task in I.TASKS:
        out[task] = next(I.iter_rows(task))
    return out


def test_infbench_prompt_is_the_official_one(first_rows, cfg):
    for task, row in first_rows.items():
        system, before, after = I.prompt_parts(row, task, cfg)
        msgs, prompt = off_i().create_msgs(None, json.loads(json.dumps(row)), task, ".", model_name=cfg["infbench"]["template_set"])
        assert before + row["context"] + after == msgs[1]["content"] == prompt, task
        assert system == msgs[0]["content"]


def test_infbench_scorer_hand_cases():
    s = I.score
    assert s("The pass key is 71432.", {"task": "passkey", "label": ["71432"]}) == 1.0
    assert s("71433", {"task": "passkey", "label": ["71432"]}) == 0.0
    assert s("3000077442", {"task": "number_string", "label": ["3000077442"]}) == 1.0
    assert s("300007744", {"task": "number_string", "label": ["3000077442"]}) == 0.0
    kv = {"task": "kv_retrieval", "label": ["5e6b7b90-710d-4953-9b18-3e96b2cadbf2"]}
    assert s('"5e6b7b90-710d-4953-9b18-3e96b2cadbf2"', kv) == 1.0 and s("5e6b7b90", kv) == 0.0
    mc = {"task": "longbook_choice_eng", "label": ["Walking Georgie", "A"]}
    assert s("A", mc) == 1.0 and s("B. Taking care of Totty", mc) == 0.0 and s("The answer is A.", mc) == 1.0


def test_infbench_evidence(first_rows):
    for task, row in first_rows.items():
        spans = I.evidence_spans(row, task)
        if task == "longbook_choice_eng":
            assert spans == []
            continue
        assert len(spans) == 1, task
        a, b = spans[0]
        ans = row["answer"][0]
        assert ans in row["context"][a:b]
