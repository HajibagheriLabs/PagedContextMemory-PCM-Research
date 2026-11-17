"""Index units, BM25, fusion, budget fill, oracle order, text expansion, index cache (CPU)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from pcm.chunking import cut_document, sentence_end_tokens, unit_bounds
from pcm.expand import expand_text
from pcm.index import select as S
from pcm.index.bm25 import BM25, terms
from pcm.index.ingest import IndexCache, doc_hash, ingest_document
from pcm.tasks.common import get_tokenizer, load_task_config


# ---------------------------------------------------------------- index units
def test_unit_bounds_without_sentences_are_multiples():
    b = unit_bounds(1000, [], unit=128, snap=32)
    assert b.tolist() == [0, 128, 256, 384, 512, 640, 768, 896, 1000]
    assert unit_bounds(100, []).tolist() == [0, 100]          # shorter than one unit: one unit
    assert unit_bounds(0, []).tolist() == [0]
    assert unit_bounds(128 * 3 + 10, []).tolist() == [0, 128, 256, 394]  # a last piece under `snap` tokens joins the unit before
    assert unit_bounds(128 * 3 + 32, []).tolist() == [0, 128, 256, 384, 416]


def test_unit_bounds_snap_to_nearest_sentence_end():
    # 120 is 8 from 128; 290 is 34 from 256 (out of reach); 380 and 388 are both 4 from 384: the earlier wins.
    b = unit_bounds(600, [120, 290, 380, 388, 500], unit=128, snap=32)
    assert b.tolist() == [0, 120, 256, 380, 500, 600]
    sizes = np.diff(b)
    assert sizes.min() >= 128 - 2 * 32 and sizes.max() <= 128 + 2 * 32 + 31
    with pytest.raises(ValueError):
        unit_bounds(1000, [], unit=64, snap=32)  # snapped boundaries could cross


def test_cut_document_covers_the_text_and_ends_units_at_sentences():
    cfg = load_task_config()
    tok = get_tokenizer(cfg["tokenizer"])
    text = " ".join(f"Sentence number {i} is here and it says a few plain words about nothing." for i in range(400))
    u = cut_document(tok, text, 128, 32)
    assert u.bounds[0] == 0 and u.bounds[-1] == len(u.ids) == u.n_tokens
    assert np.all(np.diff(u.bounds) > 0) and u.sizes.sum() == u.n_tokens
    ends = set(sentence_end_tokens(text, u.starts).tolist())
    inner = u.bounds[1:-1]
    assert all(int(b) in ends for b in inner)  # a sentence ends every ~16 tokens here: every boundary can snap
    texts = u.texts(text)
    assert len(texts) == u.n_units and all(t.strip().endswith(".") for t in texts)
    assert "".join(text[u.char_span(i)[0]:u.char_span(i)[1]] for i in range(u.n_units)).replace(" ", "") == text.replace(" ", "")


# ---------------------------------------------------------------- BM25
def test_bm25_matches_rank_bm25():
    from rank_bm25 import BM25Okapi

    docs = ["Mary went to the kitchen.", "John moved to the garden and the hallway.", "The apple is in the kitchen, the kitchen is warm.",
            "Sandra took the milk there.", "the the the", "Daniel journeyed to the office with the apple."]
    corpus = [terms(d) for d in docs]
    mine, ref = BM25(corpus), BM25Okapi(corpus)
    for q in ("Where is the apple?", "kitchen kitchen", "nobody mentioned this", "the"):
        np.testing.assert_allclose(mine.scores(q), ref.get_scores(terms(q)), rtol=1e-12, atol=1e-12)
    assert mine.idf["the"] > 0 and mine.idf["the"] < mine.idf["apple"]  # a negative IDF was replaced by the floor


def test_terms_are_lowercase_word_runs():
    assert terms('Key: "1ec0-A2" gilded-glacier?') == ["key", "1ec0", "a2", "gilded", "glacier"]


# ---------------------------------------------------------------- rankings and fusion
def test_ranks_ties_and_fusion():
    s = np.array([0.5, 0.9, 0.5, 0.0, 0.0])
    assert S.competition_ranks(s).tolist() == [2, 1, 2, 4, 4]
    assert S.order_by_score(s).tolist() == [1, 0, 2, 3, 4]  # ties in document order
    a = np.array([3.0, 2.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 5.0, 0.0])   # three units tied at 0: they share rank 2
    f = S.rrf([a, b], k=60)
    np.testing.assert_allclose(f, [1 / 61 + 1 / 62, 1 / 62 + 1 / 62, 1 / 63 + 1 / 61, 1 / 64 + 1 / 62])
    assert S.order_by_score(f).tolist() == [0, 2, 1, 3]
    with pytest.raises(ValueError):
        S.rrf([a, b[:3]])


def test_fill_budget_is_a_prefix_and_reranking_the_head_keeps_the_set():
    sizes = np.array([100, 150, 120, 90, 130, 128, 128, 128])
    order = np.array([5, 2, 7, 0, 1, 3, 4, 6])
    kept = S.fill_budget(order, sizes, 500)
    assert kept.tolist() == [5, 2, 7, 0]                       # 128 + 120 + 128 + 100 = 476; the next (150) does not fit
    assert S.fill_budget(order, sizes, 127).tolist() == []
    assert S.fill_budget(order, sizes, 10**6).tolist() == order.tolist()
    head = S.rerank_head(order, np.array([0.1, 0.9, 0.5]))     # re-order the top 3
    assert head.tolist() == [2, 7, 5, 0, 1, 3, 4, 6]
    assert set(S.fill_budget(head, sizes, 500).tolist()) == set(kept.tolist())  # same units once the head fits the budget


def test_spans_units_and_item_ranks():
    bounds = np.array([0, 100, 220, 350, 480, 600])
    assert S.spans_of([3, 0, 1], bounds) == [[0, 220], [350, 480]]       # adjacent units merge; document order
    assert S.units_of_span(95, 105, bounds) == [0, 1] and S.units_of_span(100, 220, bounds) == [1]
    ev = [{"tok_start": 95, "tok_end": 105}, {"tok_start": 500, "tok_end": 520}]
    items = S.evidence_units(ev, bounds)
    assert items == [[0, 1], [4]]
    assert S.item_ranks(np.array([4, 1, 3, 0, 2]), items) == [4, 1]      # the first item needs units 0 and 1: complete at rank 4
    packed = S.pack_units([0, 3, 4], 5)
    assert S.unpack_units(packed, 5).tolist() == [0, 3, 4]


def test_oracle_order_and_token_windows():
    order = S.oracle_order([2, 7], 10)
    assert order[:2].tolist() == [2, 7]                                   # gold first, in document order
    assert order.tolist() == [2, 7, 1, 3, 6, 8, 0, 4, 5, 9]              # then by distance, the earlier unit first
    assert S.oracle_order([2, 7], 10, pad_units=1).tolist() == [2, 7, 1, 3, 6, 8]
    assert S.oracle_order([0], 4, pad_units=0).tolist() == [0]
    with pytest.raises(ValueError):
        S.oracle_order([], 4)
    ev = [{"tok_start": 100, "tok_end": 110}, {"tok_start": 300, "tok_end": 310}]
    spans, margin = S.token_windows(ev, 1000, 120)
    assert margin == 25 and spans == [[75, 135], [275, 335]]
    spans, margin = S.token_windows(ev, 1000, 5000)
    assert spans == [[0, 1000]]                                           # room for everything
    with pytest.raises(ValueError):
        S.token_windows(ev, 1000, 19)


# ---------------------------------------------------------------- text expansion
def test_expand_text_orders_spans_and_marks_gaps():
    ctx = torch.arange(100, 200)
    gap = torch.tensor([7, 8])
    ids, gaps = expand_text(ctx, [[50, 60], [0, 10], [10, 20]], gap)
    assert gaps == 1 and ids.tolist() == list(range(100, 120)) + [7, 8] + list(range(150, 160))  # touching spans: no marker
    ids, gaps = expand_text(ctx, [[0, 100]], gap)
    assert gaps == 0 and torch.equal(ids, ctx)
    assert expand_text(ctx, [], gap)[0].numel() == 0
    with pytest.raises(ValueError):
        expand_text(ctx, [[0, 20], [10, 30]], gap)
    with pytest.raises(ValueError):
        expand_text(ctx, [[90, 101]], gap)


# ---------------------------------------------------------------- index cache
class _FakeEmbedder:
    tag = "emb-fake"
    calls = 0

    def encode_documents(self, texts):
        _FakeEmbedder.calls += 1
        rng = np.random.default_rng(len(texts))
        v = rng.normal(size=(len(texts), 16)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_index_cache_round_trip(tmp_path):
    cfg = load_task_config()
    tok = get_tokenizer(cfg["tokenizer"])
    text = " ".join(f"Plain sentence {i} about nothing in particular." for i in range(120))
    u = cut_document(tok, text, 128, 32)
    cache = IndexCache(tmp_path, "tok-x", _FakeEmbedder.tag, 128, 32)
    _FakeEmbedder.calls = 0
    first, fresh = ingest_document(u, text, _FakeEmbedder(), cache, {"example": "e1"})
    again, fresh2 = ingest_document(u, text, _FakeEmbedder(), cache)
    assert fresh and not fresh2 and _FakeEmbedder.calls == 1
    assert first.shape == (u.n_units, 16) and np.array_equal(first, again)   # a fresh and a cached run see the same float16 values
    assert cache.has(doc_hash(u.ids)) and list(cache.log.rows())[0]["units"] == u.n_units
    other = cut_document(tok, text, 128, 0)                                    # same tokens, cut differently
    if not np.array_equal(other.bounds, u.bounds):
        with pytest.raises(ValueError):
            cache.load(doc_hash(other.ids), other.bounds)
    assert doc_hash(u.ids) != doc_hash(u.ids[:-1])
