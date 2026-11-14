"""Scorers on hand-written cases, bootstrap, ECL90 on toy curves, evidence recall (CPU only)."""
from __future__ import annotations

import pytest

from pcm.eval import metrics as M


def test_contains_all():
    assert M.score_contains_all("The number is 1234567.", ["1234567"]) == 1.0
    assert M.score_contains_all("12345678", ["1234567"]) == 0.0  # digits must not touch other digits
    assert M.score_contains_all("1234567, 2345678 and 3456789", ["1234567", "2345678", "3456789"]) == 1.0
    assert M.score_contains_all("1234567, 2345678", ["1234567", "2345678", "3456789"]) == 0.0
    assert M.score_contains_all("Obelisk", ["obelisk"]) == 1.0
    assert M.score_contains_all("obelisks", ["obelisk"]) == 0.0
    assert M.score_contains_all("anything", []) == 0.0


def test_contains_exclusive():
    assert M.score_contains_exclusive("1234567", ["1234567"], ["2345678"]) == 1.0
    assert M.score_contains_exclusive("1234567 or 2345678", ["1234567"], ["2345678"]) == 0.0
    assert M.score_contains_exclusive("zither", ["obelisk"], ["zither"]) == 0.0
    assert M.score_contains_exclusive("The answer: obelisk.", ["obelisk"], ["zither", "parapet"]) == 1.0


def test_first_int():
    assert M.score_first_int("245", ["245"]) == 1.0
    assert M.score_first_int("It rang 245 times, not 3.", ["245"]) == 1.0
    assert M.score_first_int("3 or 245", ["245"]) == 0.0
    assert M.score_first_int("1,245", ["1245"]) == 1.0
    assert M.score_first_int("none", ["0"]) == 0.0


def test_synth_dispatch():
    ex = {"benchmark": "synth", "scoring": "contains_exclusive", "answers": ["1"], "exclude": ["2"]}
    assert M.score("1", ex) == 1.0 and M.score("1 2", ex) == 0.0
    with pytest.raises(ValueError):
        M.score("x", {"benchmark": "synth", "scoring": "nope", "answers": []})


def test_bootstrap_ci():
    c = M.bootstrap_ci([1.0] * 50 + [0.0] * 50, n_resamples=10_000, seed=0)
    assert c["mean"] == 0.5 and c["lo"] < 0.5 < c["hi"]
    assert 0.38 < c["lo"] < 0.42 and 0.58 < c["hi"] < 0.62  # normal approximation: 0.5 +- 0.098
    assert M.bootstrap_ci([1.0] * 10) == {"n": 10, "mean": 1.0, "lo": 1.0, "hi": 1.0}
    assert M.bootstrap_ci([0, 1, 1, 0, 1], seed=3) == M.bootstrap_ci([0, 1, 1, 0, 1], seed=3)
    assert M.bootstrap_ci([])["mean"] is None


def test_paired_bootstrap():
    a = [1.0] * 30 + [0.0] * 70
    b = [0.0] * 100
    p = M.paired_bootstrap(a, b)
    assert p["diff"] == pytest.approx(0.3) and p["excludes_0"] and p["lo"] > 0 and p["p_le_0"] == 0.0
    same = M.paired_bootstrap(a, a)
    assert same["diff"] == 0.0 and same["lo"] == same["hi"] == 0.0 and not same["excludes_0"]
    with pytest.raises(ValueError):
        M.paired_bootstrap([1.0], [1.0, 0.0])
    xa, xb, ids = M.paired_by_id([{"example_id": "b", "score": 1}, {"example_id": "a", "score": 0}],
                                 [{"example_id": "a", "score": 1}, {"example_id": "c", "score": 1}])
    assert ids == ["a"] and xa == [0.0] and xb == [1.0]


GRID = ["8k", "16k", "32k", "64k", "128k", "256k"]


def test_ecl90_toy_curves():
    # Reference 0.80: threshold 0.72, floor 0.30 (chance 0).
    curve = {"8k": 0.80, "16k": 0.78, "32k": 0.75, "64k": 0.72, "128k": 0.60, "256k": 0.75}
    e = M.ecl90(curve, 0.80, 0.0, GRID)
    assert e["ecl"] == "64k" and e["stopped_at"] == {"length": "128k", "reason": "below threshold"}
    assert e["threshold"] == pytest.approx(0.72)
    # A later recovery does not count: "at every length up to L".
    assert M.ecl90({**curve, "128k": 0.73}, 0.80, 0.0, GRID)["ecl"] == "256k"
    # Floor: reference 0.30 gives threshold 0.27, but accuracy must be above max(0.3, 2 x chance).
    low = {L: 0.30 for L in GRID}
    assert M.ecl90(low, 0.30, 0.0, GRID)["ecl"] is None
    assert M.ecl90({L: 0.45 for L in GRID}, 0.45, 0.25, GRID)["ecl"] is None  # floor 0.5 for 4 options
    assert M.ecl90({L: 0.55 for L in GRID}, 0.55, 0.25, GRID)["ecl"] == "256k"
    # Unmeasured length ends the scan.
    e = M.ecl90({"8k": 0.9, "16k": 0.9, "64k": 0.9}, 0.9, 0.0, GRID)
    assert e["ecl"] == "16k" and e["stopped_at"]["reason"] == "not measured"
    assert M.ecl_floor(0.2) == pytest.approx(0.4) and M.ecl_floor(0.0) == 0.3


def test_evidence_recall_and_units():
    ev = [{"tok_start": 10, "tok_end": 20}, {"tok_start": 100, "tok_end": 110}, {"tok_start": 500, "tok_end": 505}]
    assert M.evidence_item_recall(ev, [[0, 512]]) == 1.0
    assert M.evidence_item_recall(ev, [[0, 50], [90, 105]]) == pytest.approx(1 / 3)  # the second item is cut
    assert M.evidence_item_recall(ev, [[0, 50], [50, 120]]) == pytest.approx(2 / 3)  # adjacent ranges merge
    assert M.evidence_item_recall([], [[0, 1]]) is None
    bounds = [0, 128, 256, 384, 512]
    gold = M.units_overlapping([(10, 20), (120, 130), (500, 505)], bounds)
    assert gold == {0, 1, 3}
    rp = M.unit_recall_precision(gold, {0, 1, 2})
    assert rp["recall"] == pytest.approx(2 / 3) and rp["precision"] == pytest.approx(2 / 3)
    assert M.units_overlapping([(128, 256)], bounds) == {1}


def test_accuracy_by_decile():
    rows = [{"depth_decile": d % 3, "score": float(d % 2)} for d in range(30)] + [{"depth_decile": None, "score": 1.0}]
    by = M.accuracy_by_decile(rows)
    assert sorted(by) == [0, 1, 2] and all(by[d]["n"] == 10 for d in by)
    assert by[0]["mean"] == pytest.approx(0.5)


def test_error_kind():
    s1 = {"benchmark": "synth", "task": "synth:S1", "answers": ["1234567"], "score": 0.0}
    assert M.error_kind("7654321", s1) == "wrong"
    assert M.error_kind("I cannot find it.", s1) == "format"
    assert M.error_kind("1234567", {**s1, "score": 1.0}) == "correct"
    fr = {"benchmark": "synth", "task": "synth:S4_freq", "answers": ["obelisk"], "exclude": ["zither"], "score": 0.0}
    assert M.error_kind("zither", fr) == "wrong" and M.error_kind("bell", fr) == "format"
    cnt = {"benchmark": "synth", "task": "synth:S4_count", "answers": ["8"], "score": 0.0}
    assert M.error_kind("7", cnt) == "wrong" and M.error_kind("several", cnt) == "format"
    b = {"benchmark": "babilong", "task": "babilong:qa1", "question": "Where is Mary?", "answers": ["bathroom"], "score": 0.0}
    assert M.error_kind("The most recent location of Mary is kitchen.", b) == "wrong"
    assert M.error_kind("Mary is somewhere.", b) == "format"
    ib = {"benchmark": "infbench", "task": "infbench:longbook_choice_eng", "label": ["x", "A"], "answers": ["x"], "score": 0.0}
    assert M.error_kind("B", ib) == "wrong" and M.error_kind("I do not know", ib) == "format"
