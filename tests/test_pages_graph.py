"""Pages, the graph and PCM text mode (A5).

CPU tests: page arithmetic, the entity extractors, the three edge types, the transition
matrix, PPR against a dense reference, entity hops, the per-document graph cache, and the
PCM method through the runner with the stand-ins of tests/test_a4_methods.py (a reader that
answers with its prompt, a bag-of-words embedder). One GPU test checks PPR on the GPU
against the CPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from pcm.chunking import page_bounds, page_of_unit, units_per_page
from pcm.eval import runner as R
from pcm.index import graph as G
from pcm.index import select as S
from pcm.index.ingest import IndexCache
from pcm.runlog import JsonlLog

from test_a4_methods import UNITS, _BagEmbedder, _EchoMixin, _need_gpu, _stub_reader, cfg, examples  # noqa: F401


# ---------------------------------------------------------------- pages
def test_units_per_page_and_page_bounds():
    assert units_per_page(128) == 1 and units_per_page(512) == 4 and units_per_page(8192) == 64
    for bad in (64, 200, 0):
        with pytest.raises(ValueError):
            units_per_page(bad)
    b = np.array([0, 120, 260, 384, 500, 640, 700])  # 6 units
    assert page_of_unit(6, 4).tolist() == [0, 0, 0, 0, 1, 1]
    assert page_bounds(b, 4).tolist() == [0, 500, 700]          # the last page holds two units
    assert page_bounds(b, 1).tolist() == b.tolist() and page_bounds(b, 10).tolist() == [0, 700]


def test_page_order_fill_and_units():
    order = np.array([9, 2, 8, 3, 0, 1, 4, 5, 6, 7])           # units, best first; pages of 4: 0..3, 4..7, 8..9
    pages = S.page_order(order, 4)
    assert pages.tolist() == [2, 0, 1]                          # page of the best unit first
    assert S.fill_pages(pages, 512, 1024).tolist() == [2, 0]    # nominal count: 1024 // 512 = 2 pages
    assert S.fill_pages(pages, 512, 511).tolist() == []
    assert S.units_of_pages([2, 0], 4, 10).tolist() == [0, 1, 2, 3, 8, 9]
    assert S.item_page_ranks(pages, [[9], [3, 4], []], 4) == [1, 3, 0]
    assert S.page_order(order, 1).tolist() == order.tolist()   # pages of one unit: the unit ranking itself


# ---------------------------------------------------------------- entities
def test_sentence_initial():
    t = 'Mary went home. "John left," she said. Then\nBill came; Sue (and "Ann") too.'
    at = {w: t.index(w) for w in ("Mary", "John", "Then", "Bill", "Sue", "Ann")}
    assert G.sentence_initial(t, at["Mary"]) and G.sentence_initial(t, at["John"]) and G.sentence_initial(t, at["Then"])
    assert G.sentence_initial(t, at["Bill"])                 # after a line break
    assert not G.sentence_initial(t, at["Sue"]) and not G.sentence_initial(t, at["Ann"])


def test_surface_entities():
    t = 'Mary went to the kitchen with John. The value of amber-saddle is 2283060. He wrote "Blue Note" on key a8de and x_1.'
    e = G.surface_entities(t)
    assert "john" in e and "mary" not in e and "the" not in e and "he" not in e   # Mary and The begin sentences
    assert {"amber-saddle", "2283060", "blue note", "a8de", "x_1"} <= e
    assert "kitchen" not in e


def test_rare_terms_use_the_decile_of_occurrences():
    common = ["the", "a", "of"] * 30
    units = [common + ["kitchen"], common + ["kitchen"], common + ["garden"], common]
    rare, idf = G.rare_terms(units)
    assert rare == {"kitchen", "garden"} and idf["garden"] > idf["kitchen"] > idf["the"] == 0.0


def test_entities_incidence_idf_and_ent_edges():
    sets = [{"x", "y"}, {"x"}, {"y", "z"}, {"z"}, set()]
    ents = G.Entities.from_sets(sets)
    assert ents.vocab == ["x", "y", "z"] and ents.n_units == 5
    assert sorted(ents.vocab[i] for i in ents.of_unit(0)) == ["x", "y"] and len(ents.of_unit(4)) == 0
    assert np.allclose(ents.idf, np.log(5 / np.array([2, 2, 2])))
    r, c, w = G.ent_edges(ents, cap=20)
    edges = {(int(a), int(b)): float(v) for a, b, v in zip(r, c, w)}
    lx = np.log(2.5)
    assert edges == pytest.approx({(0, 1): lx, (1, 0): lx, (0, 2): lx, (2, 0): lx, (2, 3): lx, (3, 2): lx})
    r1, c1, _ = G.ent_edges(ents, cap=1)                       # the cap keeps the heaviest, nearest first on a tie
    assert dict(zip(r1.tolist(), c1.tolist())) == {0: 1, 1: 0, 2: 3, 3: 2}


def test_default_extractor_and_question_entities():
    texts = ["Mary went to the kitchen. She met John there.", "John picked up the football.", "The weather was fine and the sun was out."]
    ents = G.extract(texts, "default")
    assert "john" in {ents.vocab[i] for i in ents.of_unit(0)}       # "She met John there.": not sentence-initial
    assert "the" not in ents.vocab
    q = G.question_entities("Where did Mary see John?", ents, "default")
    assert ents.index["john"] in q


def test_spacy_extractor_finds_names():
    pytest.importorskip("spacy")
    ents = G.extract(["Mary went to the kitchen.", "John picked up the football."], "spacy")
    assert {"mary", "john"} <= set(ents.vocab)
    assert ents.index["mary"] in G.question_entities("Where is Mary?", ents, "spacy")


# ---------------------------------------------------------------- edges, transition, traversals
def test_sem_edges_knn_threshold():
    e = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    r, c, w = G.sem_edges(e, k=1, min_cos=0.5, device="cpu")
    assert dict(zip(r.tolist(), c.tolist())) == {0: 1, 1: 0}   # units 2 and 3 have no neighbour above 0.5
    assert np.all(w >= 0.5)


def test_transition_averages_row_normalised_types():
    g = G.Graph(3, {"ent": (np.array([0, 0]), np.array([1, 2]), np.array([10.0, 30.0]))})
    p = g.transition(["seq", "ent"]).toarray()
    # Row 0: SEQ goes to 1; ENT goes to 1 (1/4) and 2 (3/4); averaged over the two types.
    assert np.allclose(p[0], [0, 0.5 * 1 + 0.5 * 0.25, 0.5 * 0.75])
    assert np.allclose(p[1], [0.5, 0, 0.5]) and np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(G.Graph(3, {"ent": (np.zeros(0, int), np.zeros(0, int), np.zeros(0))}).transition(["ent"]).toarray(), 0)
    with pytest.raises(ValueError):
        g.transition(["nope"])


def _dense_ppr(p: np.ndarray, seeds, r=0.5, it=10):
    n = len(p)
    s = np.zeros(n)
    s[list(seeds)] = 1 / len(seeds)
    dangling = p.sum(axis=1) == 0
    x = s.copy()
    for _ in range(it):
        x = r * s + (1 - r) * (p.T @ x + x[dangling].sum() * s)
    return x


def test_ppr_matches_a_dense_reference():
    rng = np.random.default_rng(0)
    n = 40
    rows = rng.integers(0, n, 120)
    cols = rng.integers(0, n, 120)
    g = G.Graph(n, {"ent": (rows, cols, rng.random(120) + 0.1)})
    t = g.transition(["seq", "ent"])
    got = G.ppr(t, [3, 7], 0.5, 10, device="cpu")
    ref = _dense_ppr(t.toarray(), [3, 7])
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-15) and got.sum() == pytest.approx(1.0)
    # A node no path reaches from the seeds scores 0.
    t2 = G.Graph(4, {"ent": (np.array([0]), np.array([1]), np.array([1.0]))}).transition(["ent"])
    assert G.ppr(t2, [0], device="cpu")[3] == 0.0


def test_entity_hops_follow_shared_entities():
    # Unit 0 mentions the question's entity "ball" and "john"; unit 5 says where John went; unit 9 is unrelated.
    sets = [{"ball", "john"}, set(), {"weather"}, set(), set(), {"john", "kitchen"}, set(), {"kitchen", "mary"}, set(), {"sun"}]
    ents = G.Entities.from_sets(sets)
    base = np.array([0, 9, 2, 1, 3, 4, 6, 8, 5, 7])
    order, added = G.entity_hops(base, ents, ents.ids(["ball"]), hops=1, seeds=1, hop_k=2)
    assert order[:2].tolist() == [0, 5] and added == [1]       # one hop: John's other unit joins the seed
    order2, added2 = G.entity_hops(base, ents, ents.ids(["ball"]), hops=2, seeds=1, hop_k=2)
    assert order2[:3].tolist() == [0, 5, 7] and added2 == [1, 1]  # second hop: through "kitchen"
    assert sorted(order2.tolist()) == list(range(10))
    none, a0 = G.entity_hops(base, G.Entities.from_sets([set()] * 10), [], hops=2, seeds=1)
    assert none.tolist() == base.tolist() and a0 == [0, 0]


def test_graph_cache_roundtrip(tmp_path):
    ents = G.Entities.from_sets([{"a", "b"}, {"b"}], {"rare": ["b"]})
    edges = G.ent_edges(ents)
    cache = G.GraphCache(tmp_path)
    assert cache.load("ent-default-c20", "ab12") is None
    size = cache.save("ent-default-c20", "ab12", G.pack_entities(ents, edges))
    back, e2 = G.unpack_entities(cache.load("ent-default-c20", "ab12"))
    assert size > 0 and back.vocab == ents.vocab and back.extra["rare"] == ["b"] and np.array_equal(back.indptr, ents.indptr)
    assert all(np.array_equal(x, y) for x, y in zip(e2, edges))


# ---------------------------------------------------------------- PCM text mode through the runner
class _EchoPCM(_EchoMixin, R.PCMText):
    cache_root = None

    def setup(self):
        self.check_config()
        self.setup_ranker()
        _stub_reader(self)
        self.model = None
        self.setup_units()
        self.embedder = _BagEmbedder()
        self.cache = IndexCache(type(self).cache_root, "tok-test", self.embedder.tag, self.unit, self.snap)
        self.setup_pages()


@pytest.fixture()
def echo_pcm(monkeypatch, tmp_path):
    _EchoPCM.cache_root = tmp_path / "index"
    _EchoPCM.generated = 0
    monkeypatch.setitem(R.METHODS, "pcm", _EchoPCM)


def _run(mc, tmp_path, examples, cfg, exp="t", **kw):
    return R.run(mc, "synth:mixed", "4k", None, exp=exp, results_root=tmp_path, examples=examples, allow_dirty=True, cfg=cfg,
                 log=lambda m: None, **kw)


BASE = {"method": "pcm", "reader": "reader", **UNITS, "ranker": "hybrid", "rrf_k": 60, "bm25": {"k1": 1.5, "b": 0.75, "epsilon": 0.25}}


def test_pcm_pages_fill_and_document_order(tmp_path, cfg, examples, echo_pcm):
    for page in (128, 256, 512):
        assert _run({**BASE, "name": f"p{page}", "page": page}, tmp_path, examples, cfg)["ok"] == len(examples)
    rows = list(JsonlLog(tmp_path / "t" / "runs.jsonl").rows())
    for r in rows:
        sel = r["selection"]
        per = sel["page"] // 128
        kept = S.unpack_units(sel["units_b64"], sel["n_units"])
        assert sel["kept_pages"] == UNITS["budget"] // sel["page"]            # nominal page count
        assert len(kept) <= sel["kept_pages"] * per and len(set((kept // per).tolist())) == sel["kept_pages"]
        assert all(set(range(p * per, min((p + 1) * per, sel["n_units"]))) <= set(kept.tolist()) for p in set(kept // per))  # whole pages
        assert r["score"] == float(r["evidence_recall"] == 1.0)              # the echo reader: right iff the evidence is in
        assert len(sel["evidence_page_ranks"]) == len(r["answers"])
        recalled = sum(1 for k in sel["evidence_page_ranks"] if k <= sel["kept_pages"])
        assert r["evidence_recall"] == recalled / len(sel["evidence_page_ranks"])
    # Page 128 keeps the budget's nominal number of units of the hybrid ranking.
    p128 = [r for r in rows if r["method"] == "p128"]
    assert all(r["selection"]["kept_units"] == UNITS["budget"] // 128 for r in p128)


def test_pcm_graph_variants_cache_and_reuse(tmp_path, cfg, examples, echo_pcm):
    variants = {"none": None, "seq": {"traversal": "ppr", "edges": ["seq"]}, "all": {"traversal": "ppr", "edges": ["seq", "sem", "ent"]},
                "hops1": {"traversal": "hops", "hops": 1}, "hops2": {"traversal": "hops", "hops": 2, "seeds": 2, "hop_k": 2}}
    for name, g in variants.items():
        mc = {**BASE, "name": name, "page": 256, **({"graph": g} if g else {})}
        assert _run(mc, tmp_path, examples, cfg)["ok"] == len(examples)
    rows = list(JsonlLog(tmp_path / "t" / "runs.jsonl").rows())
    by = {n: [r for r in rows if r["method"] == n] for n in variants}
    assert all("graph" not in r["selection"] for r in by["none"])
    assert all(r["selection"]["graph"]["traversal"] == "ppr" and r["selection"]["graph"]["reached"] > 0 for r in by["all"])
    assert all(len(r["selection"]["graph"]["added"]) == 2 for r in by["hops2"])
    # Entities and SEM edges were built once per document and then read from the cache.
    assert sum(1 for r in rows if "ent_build_s" in r["timings"]) == len(examples)
    assert sum(1 for r in rows if "sem_build_s" in r["timings"]) == len(examples)
    assert any((tmp_path / "index").rglob("*.npz"))
    # Identical prompts are answered once.
    assert sum(1 for r in rows if r["reused_from"] is None) == _EchoPCM.generated
    with pytest.raises(ValueError):
        R.PCMText({**BASE, "page": 256, "graph": {"traversal": "ppr"}}).check_config()
    with pytest.raises(ValueError):
        R.PCMText({**BASE, "page": 256, "graph": {"traversal": "hops", "hops": 0}}).check_config()


def test_pcm_selection_only_rows_hold_no_answer(tmp_path, cfg, examples, echo_pcm):
    sel_only = {**BASE, "name": "sel", "page": 256, "selection_only": True}
    assert _run(sel_only, tmp_path, examples, cfg)["ok"] == len(examples)
    assert _EchoPCM.generated == 0
    rows = list(JsonlLog(tmp_path / "t" / "runs.jsonl").rows())
    assert all(r["prediction"] is None and r["score"] is None and r["correct"] is None and r["selection_only"] for r in rows)
    assert all(r["evidence_recall"] is not None for r in rows)
    # A method with the reader and the same prompts must not take the missing answers over.
    _run({**BASE, "name": "full", "page": 256}, tmp_path, examples, cfg)
    full = [r for r in JsonlLog(tmp_path / "t" / "runs.jsonl").rows() if r["method"] == "full"]
    assert _EchoPCM.generated == len(examples) and all(r["reused_from"] is None and r["prediction"] for r in full)


def test_pcm_reuses_answers_of_another_experiment(tmp_path, cfg, examples, echo_pcm):
    mc = {**BASE, "name": "p256", "page": 256}
    _run(mc, tmp_path, examples, cfg, exp="first")
    n = _EchoPCM.generated
    _run({**mc, "name": "p256-again"}, tmp_path, examples, cfg, exp="second", answer_from=["first"])
    rows = list(JsonlLog(tmp_path / "second" / "runs.jsonl").rows())
    assert _EchoPCM.generated == n and all(r["reused_from"] and r["timings"]["reused"] for r in rows)


def test_pcm_rejects_a_page_larger_than_the_budget(tmp_path, cfg, examples, echo_pcm):
    with pytest.raises(ValueError, match="does not fit the budget"):
        _run({**BASE, "name": "big", "page": 1024}, tmp_path, examples[:1], cfg, exp="big")  # budget 640


@pytest.mark.gpu
def test_ppr_on_gpu_matches_cpu(request):
    _need_gpu(request)
    rng = np.random.default_rng(1)
    n = 500
    g = G.Graph(n, {"ent": (rng.integers(0, n, 3000), rng.integers(0, n, 3000), rng.random(3000)),
                    "sem": (rng.integers(0, n, 2500), rng.integers(0, n, 2500), rng.random(2500))})
    t = g.transition(["seq", "sem", "ent"])
    a, b = G.ppr(t, list(range(16)), device="cpu"), G.ppr(t, list(range(16)), device="cuda")
    assert np.allclose(a, b, rtol=1e-9, atol=1e-18)
    assert np.array_equal(S.order_by_score(a), S.order_by_score(b)) and np.array_equal(b, G.ppr(t, list(range(16)), device="cuda"))


# ---------------------------------------------------------------- run script: cells and the choices between stages
def _run_a5():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("run_a5", root / "scripts" / "run_a5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_rows(mod, cfg, page, budget, cells, score, recall=1.0, n=30, variant="none"):
    from pcm.tasks.common import config_hash

    h = config_hash(mod.setting_config(cfg, page, budget, variant))
    return [{"method_hash": h, "task": t, "length": L, "status": "ok", "score": float(i < round(score * n)), "evidence_recall": recall}
            for t, L in cells for i in range(n)]


def test_run_a5_cells():
    from pcm.tasks.common import config_hash

    mod = _run_a5()
    cfg = mod.load_config()
    settings = mod.screen_settings(cfg)
    assert len(settings) == 27 and (8192, 4096) not in settings and (4096, 4096) in settings
    screen = mod.cells_for(cfg, "screen", None)
    assert len(screen) == 27 * 2 * 4 and all(c["n"] == 30 for c in screen)
    sel = [i for i, c in enumerate(screen) if c["mcfg"].get("selection_only")]
    assert len(sel) == 27 * 2 and sel == list(range(len(screen) - len(sel), len(screen)))   # selection-only cells last
    final = mod.cells_for(cfg, "final", {"settings": [{"page": 512, "budget": 8192}, {"page": 128, "budget": 4096}, {"page": 2048, "budget": 28672}]})
    assert len(final) == 3 * (13 * 2 + 4) and {c["length"] for c in final if c["task"].startswith("infbench:")} == {"full"}
    e5 = mod.cells_for(cfg, "e5", {"page": 512, "budget": 8192})
    assert len(e5) == 7 * (8 * 2 + 4) + 4 * (5 * 2) + 3 * (5 * 2) + 3 * (13 * 2) + 3 * (3 * 2)
    none = [c for c in e5 if c["method"] == "pcm-p512-b8192"]
    assert none and config_hash(none[0]["mcfg"]) == config_hash(mod.setting_config(cfg, 512, 8192))   # same hash as E4's setting
    ner = [c for c in e5 if c["method"].endswith("-ner")]
    assert ner and all(c["mcfg"]["graph"]["extractor"] == "spacy" and c["task"].startswith("babilong:") for c in ner)
    hops = next(c for c in e5 if c["method"].endswith("-hops2"))["mcfg"]["graph"]
    assert hops == {**cfg["graph"], "extractor": "default", "traversal": "hops", "hops": 2}
    assert all(c["mcfg"]["page"] == 512 and c["mcfg"]["budget"] == 8192 for c in e5)


def test_run_a5_choices_follow_the_fixed_rules():
    mod = _run_a5()
    cfg = mod.load_config()
    sc = cfg["e4"]["screen"]
    cells = [(t, L) for L in sc["lengths"] for t in sc["tasks"]]
    rows = []
    for p, b in mod.screen_settings(cfg):
        acc, rec = 0.5, 1.0
        if (p, b) == (512, 8192):
            acc = 0.9
        elif (p, b) in ((256, 8192), (1024, 8192)):
            acc, rec = 0.8, (0.9 if p == 256 else 0.8)
        elif (p, b) in ((128, 16384), (128, 4096)):
            acc = 0.7                                     # same accuracy and recall: the smaller budget first
        rows += _fake_rows(mod, cfg, p, b, cells, acc, rec)
    best = mod.choose_best_settings(cfg, rows)
    assert best["settings"] == [{"page": 512, "budget": 8192}, {"page": 256, "budget": 8192}, {"page": 1024, "budget": 8192}]
    order = [(s["page"], s["budget"]) for s in best["ranking"]]
    assert order.index((128, 4096)) < order.index((128, 16384))
    with pytest.raises(RuntimeError, match="incomplete"):
        mod.choose_best_settings(cfg, rows[:-1])
    # E5's setting: the mean over 128K and 256K of the S1, S2 and BABILong accuracies.
    fam_tasks = [t for f in cfg["h1_families"] for t in cfg["families"][f]]
    final_cells = [(t, L) for L in cfg["e4"]["final"]["lengths"] for t in fam_tasks]
    r2 = (_fake_rows(mod, cfg, 512, 8192, final_cells, 0.6, n=100) + _fake_rows(mod, cfg, 256, 8192, final_cells, 0.7, n=100)
          + _fake_rows(mod, cfg, 1024, 8192, final_cells, 0.7, n=100))
    e5 = mod.choose_e5_setting(cfg, r2, best)
    assert (e5["page"], e5["budget"]) == (256, 8192)       # tie with 1024 at 0.7: the smaller page
