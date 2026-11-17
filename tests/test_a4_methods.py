"""Baseline methods of A4 through the runner: oracle, retrieval, dense YaRN.

CPU tests replace the reader by one that "answers" with the prompt it was given, so an
example scores 1 exactly when its evidence reached the prompt; embedder and reranker are
replaced by deterministic stand-ins. GPU tests check the embedder and the reranker against
the reference values of their model cards and run each method once on tiny examples.
"""
from __future__ import annotations

import numpy as np
import pytest

from pcm.eval import runner as R
from pcm.health import EXIT_GPU_LOST
from pcm.index import select as S
from pcm.index.bm25 import terms
from pcm.index.ingest import IndexCache
from pcm.runlog import JsonlLog
from pcm.tasks.common import get_tokenizer, load_task_config
from pcm.tasks.synth import BookPool, ensure_book_cache, generate_example

UNITS = {"unit": 128, "snap": 32, "budget": 640, "gap_marker": "\n[...]\n", "window": 32768}


@pytest.fixture(scope="module")
def cfg():
    return load_task_config()


@pytest.fixture(scope="module")
def examples(cfg):
    """S1 and S2 multi-value examples of about 4K tokens (some 32 index units each)."""
    tok = get_tokenizer(cfg["tokenizer"])
    pool = BookPool(ensure_book_cache(tok, cfg, log=lambda m: None))
    out = []
    for task in ("S1", "S2_multivalue"):
        for i in range(4):
            out.append({**generate_example(task, 4096, 0, i, tok, pool, length_name="4k", cfg=cfg), "benchmark": "synth"})
    return out


def _rows(tmp_path, exp):
    return list(JsonlLog(tmp_path / exp / "runs.jsonl").rows())


def _run(mc, tmp_path, examples, cfg, exp="t"):
    return R.run(mc, "synth:mixed", "4k", None, exp=exp, results_root=tmp_path, examples=examples, allow_dirty=True, cfg=cfg, log=lambda m: None)


# ---------------------------------------------------------------- stand-ins
def _stub_reader(method):
    tok = get_tokenizer("reader")
    method.tokenizer = method.prep_tokenizer = tok
    method.stops = [0]


class _EchoMixin:
    """No model: the answer is the prompt text."""

    needs_gpu = False
    generated = 0

    def reader_key(self):
        return ("echo", "none", None)

    def _generate(self, example, ids):
        type(self).generated += 1
        return self.tokenizer.decode(ids), {"new_tokens": 1}


class _EchoOracle(_EchoMixin, R.Oracle):
    def setup(self):
        _stub_reader(self)
        self.setup_units()


class _BagEmbedder:
    """Hashed bag of words, normalised: units that share words with the query score high."""

    tag = "emb-bag"
    query_prompt = ""
    dim = 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in terms(t):
                out[i, int.from_bytes(w.encode("utf-8")[:8].ljust(8, b"\0"), "little") % self.dim] += 1.0
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    encode_documents = encode

    def encode_query(self, q):
        return self.encode([q])[0]


class _ReverseReranker:
    """Scores the units it is given in reverse order: the head of the ranking is turned around."""

    def score(self, query, docs):
        return np.arange(len(docs), dtype=np.float32)


class _EchoRetrieve(_EchoMixin, R.Retrieve):
    cache_root = None

    def setup(self):
        self.setup_ranker()
        _stub_reader(self)
        self.setup_units()
        if self.uses_emb:
            self.embedder = _BagEmbedder()
            self.cache = IndexCache(type(self).cache_root, "tok-test", self.embedder.tag, self.unit, self.snap)
        if self.ranker == "hybrid_rerank":
            self.reranker = _ReverseReranker()


@pytest.fixture()
def echo_methods(monkeypatch, tmp_path):
    _EchoRetrieve.cache_root = tmp_path / "index"
    _EchoOracle.generated = _EchoRetrieve.generated = 0
    monkeypatch.setitem(R.METHODS, "oracle", _EchoOracle)
    monkeypatch.setitem(R.METHODS, "retrieve", _EchoRetrieve)


# ---------------------------------------------------------------- oracle (M3)
def test_oracle_keeps_all_evidence_inside_the_budget(tmp_path, cfg, examples, echo_methods):
    full = {"method": "oracle", "name": "oracle", "reader": "reader", **UNITS}
    tight = {**full, "name": "oracle-tight", "pad_units": 1}
    assert _run(full, tmp_path, examples, cfg)["ok"] == len(examples)
    assert _run(tight, tmp_path, examples, cfg)["ok"] == len(examples)
    rows = _rows(tmp_path, "t")
    for r in rows:
        sel = r["selection"]
        assert r["evidence_recall"] == 1.0 and r["score"] == 1.0 and sel["unit_recall"] == 1.0  # the evidence reached the prompt
        assert sel["kept_tokens"] <= UNITS["budget"] and r["selected_spans"] is None and sel["oracle"] == "units"
        kept = S.unpack_units(sel["units_b64"], sel["n_units"])
        assert len(kept) == sel["kept_units"] and "evidence_ranks" not in sel
    by = {m: [r for r in rows if r["method"] == m] for m in ("oracle", "oracle-tight")}
    for a, b in zip(by["oracle"], by["oracle-tight"]):
        assert b["selection"]["kept_units"] <= min(3 * b["selection"]["gold_units"], a["selection"]["kept_units"])
    s1 = [r for r in by["oracle"] if r["example_id"].startswith("S1-") and r["selection"]["gold_units"] == 1]
    assert s1 and all(r["selection"]["gaps"] == 0 and r["selection"]["kept_tokens"] > UNITS["budget"] - 256 for r in s1)  # one padded block


def test_oracle_skips_examples_without_evidence_and_falls_back_to_token_windows(tmp_path, cfg, examples, echo_methods):
    bare = [{**examples[0], "id": "no-evidence", "evidence": []}]
    s = _run({"method": "oracle", "name": "oracle", "reader": "reader", **UNITS}, tmp_path, bare, cfg, exp="skip")
    assert s["skipped"] == 1 and "no evidence" in _rows(tmp_path, "skip")[0]["reason"]
    multi = [e for e in examples if len(e["evidence"]) == 4][:1]
    s = _run({"method": "oracle", "name": "oracle", "reader": "reader", **UNITS, "budget": 200}, tmp_path, multi, cfg, exp="win")
    r = _rows(tmp_path, "win")[0]
    assert s["ok"] == 1 and r["selection"]["oracle"] == "token_windows" and r["evidence_recall"] == 1.0 and r["score"] == 1.0
    assert r["selection"]["kept_tokens"] <= 200 and r["selected_spans"]  # not whole units: the token spans are stored


# ---------------------------------------------------------------- retrieval (M2)
def test_retrieval_variants_select_rank_and_reuse(tmp_path, cfg, examples, echo_methods):
    base = {"method": "retrieve", "reader": "reader", **UNITS, "rrf_k": 60, "rerank_top": 2}
    names = {}
    for ranker in ("bm25", "embed", "hybrid", "hybrid_rerank"):
        names[ranker] = f"m2-{ranker}"
        assert _run({**base, "name": names[ranker], "ranker": ranker}, tmp_path, examples, cfg)["ok"] == len(examples)
    rows = _rows(tmp_path, "t")
    by = {k: {r["example_id"]: r for r in rows if r["method"] == v} for k, v in names.items()}
    for ranker, rs in by.items():
        for r in rs.values():
            sel = r["selection"]
            assert sel["ranker"] == ranker and sel["kept_tokens"] <= UNITS["budget"] and r["selected_spans"] is None
            assert len(sel["evidence_ranks"]) == len(r["answers"]) and min(sel["evidence_ranks"]) >= 1
            kept = S.unpack_units(sel["units_b64"], sel["n_units"])
            recalled = sum(1 for k in sel["evidence_ranks"] if k <= len(kept))
            assert r["evidence_recall"] == recalled / len(sel["evidence_ranks"])  # recall follows from the ranks and the cut
            assert r["score"] == float(r["evidence_recall"] == 1.0)               # the echo reader: right iff the evidence is in
    # The needle names its key, which the question repeats: BM25 finds the single needle of S1.
    s1 = [r for r in by["bm25"].values() if len(r["answers"]) == 1]
    assert s1 and all(r["evidence_recall"] == 1.0 and r["selection"]["evidence_ranks"][0] <= 3 for r in s1)
    # Re-ordering the top 2 units cannot change what a budget of five units keeps: same prompt, answer reused.
    by_id = {r["id"]: r for r in rows}
    for eid, r in by["hybrid_rerank"].items():
        h = by["hybrid"][eid]
        assert r["selection"]["units_b64"] == h["selection"]["units_b64"] and r["prompt_sha"] == h["prompt_sha"]
        src = by_id[r["reused_from"]]  # the first row that answered this prompt (any variant that kept the same units)
        assert src["prompt_sha"] == r["prompt_sha"] and src["example_id"] == eid and src["reused_from"] is None
        assert r["timings"]["reused"] is True and r["prediction"] == h["prediction"] == src["prediction"]
    assert sum(1 for r in rows if r["reused_from"] is None) == _EchoRetrieve.generated  # the reader ran once per distinct prompt
    assert any(r["selection"]["rerank_moved"] > 0 for r in by["hybrid_rerank"].values())
    # Embeddings were computed once per document and served from the cache afterwards.
    ingested = [r for r in rows if "ingest_s" in r["timings"]]
    assert len(ingested) == len(examples) and all(r["method"] == names["embed"] for r in ingested)


def test_retrieval_keeps_a_short_document_whole(tmp_path, cfg, examples, echo_methods):
    mc = {"method": "retrieve", "name": "m2-bm25", "reader": "reader", **UNITS, "ranker": "bm25", "budget": 10**6}
    _run(mc, tmp_path, examples[:2], cfg, exp="whole")
    for r in _rows(tmp_path, "whole"):
        sel = r["selection"]
        assert sel["kept_units"] == sel["n_units"] and sel["gaps"] == 0 and sel["kept_tokens"] == r["length_tokens"]


def test_query_text_appends_options():
    assert R.query_text({"question": " Why? ", "row": {"options": ["a", "b"]}}) == "Why?\na\nb"
    assert R.query_text({"question": "Where is Mary? "}) == "Where is Mary?"


def test_window_is_checked_after_expansion(tmp_path, cfg, examples, echo_methods):
    mc = {"method": "oracle", "name": "oracle", "reader": "reader", **UNITS, "window": 300}
    assert _run(mc, tmp_path, examples[:1], cfg, exp="w")["skipped"] == 1


# ---------------------------------------------------------------- run script: cells, plan
def test_run_script_cells_and_plan():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("run_a4", Path(__file__).resolve().parent.parent / "scripts" / "run_a4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c = mod.load_config()
    e2 = mod.cells(c, "E2_dense")
    assert len(e2) == 8 * 2 + 2 + 5 * 2 and {x["n"] for x in e2} == {"plan", "fixed"}
    assert len(mod.cells(c, "E3_retrieval")) == 4 * (13 * 4 + 4) and len(mod.cells(c, "E3o_oracle")) == 2 * (13 * 4 + 3) + 13 * 2
    assert len(mod.cells(c, "E3_retrieval", methods=["m2-bm25-1.7B"], tasks=["synth:S1"], lengths=["256k"])) == 1
    for name, m in c["methods"].items():
        assert m["name"] == name and m["reader"] == "reader" and m["max_new_tokens"]["infbench:kv_retrieval"] == 128
        assert m["max_new_tokens"]["synth:S1"] == 64 and m["max_new_tokens"]["synth:S2_multivalue"] == 128   # D-040
    ex = {"benchmark": "synth", "task": "S1", "max_new_tokens": 16}
    assert R.Native(c["methods"]["m2-hybrid-1.7B"]).max_new(ex) == 64 and R.Native({"reader": "reader"}).max_new(ex) == 16
    assert len(mod.cells(c, "E0_a4budget")) == 3
    assert c["methods"]["m2-hybrid-1.7B"]["budget"] == 28672 and c["methods"]["m3-oracle-tight-1.7B"]["pad_units"] == 1
    # Plan: 2 `plan` cells at 20 s and 80 s, 1 fixed cell at 80 s, cap 1 hour.
    cfg = {"e2": {"cap_hours": 1.0, "n_fixed": 30, "n_candidates": [100, 80, 60, 50, 40, 30]}}
    todo = [{"task": "a", "length": "64k", "n": "plan", "mcfg": {"x": 1}}, {"task": "a", "length": "128k", "n": "plan", "mcfg": {"x": 2}},
            {"task": "b", "length": "128k", "n": "fixed", "mcfg": {"x": 2}}]
    stats = {mod.cell_key(todo[0]): {"seconds": [20.0, 20.0]}, mod.cell_key(todo[1]): {"seconds": [80.0]}, mod.cell_key(todo[2]): {"seconds": [80.0]}}
    plan = mod.make_plan(todo, stats, cfg)
    assert plan["n_plan"] is None and plan["projected_hours_by_n"]["30"] == round((30 * 80 + 30 * 100) / 3600, 2)  # 1.5 h: over the cap
    cfg["e2"]["cap_hours"] = 2.0
    plan = mod.make_plan(todo, stats, cfg)
    assert plan["n_plan"] == 40 and plan["projected_hours"] == round((30 * 80 + 40 * 100) / 3600, 2)
    assert mod.n_for({"n": "plan"}, {"n": 100, "e2": cfg["e2"]}, None) is None and mod.n_for({"n": "plan"}, {"n": 100, "e2": cfg["e2"]}, plan) == 40
    assert mod.n_for({"n": "fixed"}, {"n": 100, "e2": cfg["e2"]}, None) == 30 and mod.n_for({"n": "all"}, {"n": 100, "e2": cfg["e2"]}, None) == 100


# ---------------------------------------------------------------- GPU: real models
def _need_gpu(request):
    from pcm.health import gpu_status

    if request.config.getoption("--no-gpu"):
        pytest.skip("GPU tests skipped on request (--no-gpu)")
    st = gpu_status()
    if not st["ok"]:
        pytest.exit(f"no usable GPU: {st['reason']}", returncode=EXIT_GPU_LOST)


@pytest.mark.gpu
def test_embedder_reproduces_its_model_card(request):
    """The four similarities of the embedder's model card example (0.7646, 0.1414, 0.1355, 0.6000)."""
    _need_gpu(request)
    from pcm.index.embed import Embedder

    e = Embedder("embedder")
    q = np.stack([e.encode_query("What is the capital of China?"), e.encode_query("Explain gravity")])
    d = e.encode_documents([
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ])
    sim = q @ d.T
    print("embedder similarities:", np.round(sim, 4).tolist())
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(sim, [[0.7646, 0.1414], [0.1355, 0.6000]], atol=0.02)  # bf16 against the card's float32
    one_by_one = np.stack([e.encode_documents([t])[0] for t in ("The capital of China is Beijing.", "A much longer second sentence, so that the first one is padded on the left in a batch.")])
    batched = e.encode_documents(["The capital of China is Beijing.", "A much longer second sentence, so that the first one is padded on the left in a batch."])
    assert float(np.abs(one_by_one - batched).max()) < 0.02  # padding does not change an embedding


@pytest.mark.gpu
def test_reranker_reproduces_its_model_card(request):
    """The model card's example pairs: logit differences 7.625 (relevant) and -11.375 (not relevant)."""
    _need_gpu(request)
    from pcm.index.rerank import Reranker

    r = Reranker("reranker")
    docs = ["The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."]
    s = r.score("What is the capital of China?", docs)
    print("reranker scores:", s.tolist())
    np.testing.assert_allclose(s, [7.625, -11.375], atol=1.0)
    assert r.score("What is the capital of China?", docs[::-1]).argmax() == 1  # order of the pairs does not matter


@pytest.mark.gpu
def test_methods_tiny_run_on_gpu(request, tmp_path, cfg, examples):
    """Oracle, the four retrieval variants and dense YaRN on tiny examples with the real models."""
    _need_gpu(request)
    reader = request.config.getoption("--reader")
    units = {**UNITS, "budget": 1024, "index_cache": str(tmp_path / "index")}
    few = examples[:2] + examples[4:6]
    for mc in ({"method": "oracle", "name": "oracle", "reader": reader, **units},
               {"method": "retrieve", "name": "hybrid", "reader": reader, "ranker": "hybrid", **units},
               {"method": "retrieve", "name": "hybrid-rerank", "reader": reader, "ranker": "hybrid_rerank", "rerank_top": 4, **units},
               {"method": "retrieve", "name": "bm25", "reader": reader, "ranker": "bm25", **units},
               {"method": "retrieve", "name": "embed", "reader": reader, "ranker": "embed", **units}):
        s = R.run(mc, "synth:mixed", "4k", None, exp="g", results_root=tmp_path, examples=few, allow_dirty=True, cfg=cfg, log=lambda m: None)
        assert s["ok"] == len(few), s
    rows = _rows(tmp_path, "g")
    by = {m: {r["example_id"]: r for r in rows if r["method"] == m} for m in ("oracle", "hybrid", "hybrid-rerank", "bm25", "embed")}
    assert all(r["evidence_recall"] == 1.0 for r in by["oracle"].values())
    for r in rows:
        assert r["prompt_tokens"] < 1024 + 400 and 0 < r["peak_vram_gib"] < 24 and r["selection"]["kept_tokens"] <= 1024
    assert all(r["reused_from"] == by["hybrid"][eid]["id"] for eid, r in by["hybrid-rerank"].items())
    assert sum("ingest_s" in r["timings"] for r in rows) == len(few)   # each document embedded once
    s1 = [r for m in ("hybrid", "bm25", "embed") for r in by[m].values() if len(r["answers"]) == 1]
    print("tiny retrieval:", [(r["method"], r["evidence_recall"], r["selection"]["evidence_ranks"], r["score"], r["prediction"][:30]) for r in s1])
    assert all(r["evidence_recall"] == 1.0 for r in s1)                # one needle, eight of 32 units kept
    R.release_readers()
    dense = {"method": "dense_yarn", "name": "yarn", "reader": reader, "yarn_factor": 2}
    s = R.run(dense, "synth:mixed", "4k", None, exp="g", results_root=tmp_path, examples=few[:1], allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert s["ok"] == 1
    m = R.make_method(dense)
    m.setup()
    assert m.window_size() == 2 * 40960 and m.reader_key()[2] == 2.0
    with pytest.raises(ValueError):
        R.make_method({"method": "dense_yarn", "reader": reader}).setup()
    R.release_readers()
