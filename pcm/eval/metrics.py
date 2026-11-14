"""Scorers and statistics (specification, section 5.4).

- Task scorers: synthetic tasks (rules named in configs/a2_tasks.yaml), BABILong and
  InfiniteBench (their official scorers, through pcm.tasks.babilong / infbench).
- Bootstrap confidence interval of a mean, and paired bootstrap of a difference of
  means on the same examples: percentile method, 10,000 resamples, fixed seed.
- ECL90: the largest grid length L such that, at every grid length up to L, accuracy
  is at least 90% of the reference (native accuracy at 8K) and above the floor
  max(30%, 2 x chance).
- Evidence recall and precision at budget: over evidence items (an item counts when
  it lies wholly inside the expanded token ranges) and over index units (units that
  overlap gold evidence, against the units expanded).
- Accuracy by evidence depth decile.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- task scorers
_INT = re.compile(r"-?\d+")


def _present(text: str, answer: str) -> bool:
    """Whole-number match for digit strings; case-insensitive whole-word match otherwise."""
    if answer.isdigit():
        return re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", text) is not None
    return re.search(rf"(?<![\w-]){re.escape(answer)}(?![\w-])", text, re.IGNORECASE) is not None


def score_contains_all(prediction: str, answers: Sequence[str]) -> float:
    return float(bool(answers) and all(_present(prediction, a) for a in answers))


def score_contains_exclusive(prediction: str, answers: Sequence[str], exclude: Sequence[str]) -> float:
    """Every answer present and no excluded string (a distractor's value or word) present."""
    return float(score_contains_all(prediction, answers) == 1.0 and not any(_present(prediction, x) for x in exclude))


def score_first_int(prediction: str, answers: Sequence[str]) -> float:
    m = _INT.search(prediction.replace(",", ""))
    return float(m is not None and int(m.group(0)) == int(answers[0]))


SYNTH_SCORERS = {"contains_all", "contains_exclusive", "first_int"}


def score_synth(prediction: str, example: Mapping) -> float:
    rule = example["scoring"]
    if rule == "contains_all":
        return score_contains_all(prediction, example["answers"])
    if rule == "contains_exclusive":
        return score_contains_exclusive(prediction, example["answers"], example.get("exclude", []))
    if rule == "first_int":
        return score_first_int(prediction, example["answers"])
    raise ValueError(f"unknown scoring rule {rule!r}")


def score(prediction: str, example: Mapping) -> float:
    """Score of one prediction (0.0 or 1.0) with the scorer of the example's benchmark."""
    bench = example.get("benchmark", "synth")
    if bench == "synth":
        return score_synth(prediction, example)
    if bench == "babilong":
        from pcm.tasks.babilong import score as babilong_score

        return babilong_score(prediction, example)
    if bench == "infbench":
        from pcm.tasks.infbench import score as infbench_score

        return infbench_score(prediction, example)
    raise ValueError(f"unknown benchmark {bench!r}")


# ---------------------------------------------------------------- bootstrap
def bootstrap_ci(values: Sequence[float], n_resamples: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean and percentile bootstrap interval of the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)
    a = (1 - ci) / 2
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(np.quantile(means, a)), "hi": float(np.quantile(means, 1 - a))}


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean of a - b over paired examples, its percentile interval, and the share of resamples with a - b <= 0."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("paired samples must have the same length")
    d = xa - xb
    if d.size == 0:
        return {"n": 0, "diff": None, "lo": None, "hi": None, "p_le_0": None, "excludes_0": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    diffs = d[idx].mean(axis=1)
    q = (1 - ci) / 2
    lo, hi = float(np.quantile(diffs, q)), float(np.quantile(diffs, 1 - q))
    return {"n": int(d.size), "diff": float(d.mean()), "lo": lo, "hi": hi, "p_le_0": float((diffs <= 0).mean()), "excludes_0": bool(lo > 0 or hi < 0)}


def paired_by_id(rows_a: Iterable[Mapping], rows_b: Iterable[Mapping], key: str = "example_id", value: str = "score") -> tuple[list[float], list[float], list[str]]:
    """Scores of two methods on the examples both have, in a fixed order (sorted ids)."""
    da = {r[key]: float(r[value]) for r in rows_a}
    db = {r[key]: float(r[value]) for r in rows_b}
    ids = sorted(set(da) & set(db))
    return [da[i] for i in ids], [db[i] for i in ids], ids


# ---------------------------------------------------------------- ECL90
ECL_EPS = 1e-9


def ecl_floor(chance: float, floor_min: float = 0.3) -> float:
    return max(floor_min, 2.0 * chance)


def ecl90(
    curve: Mapping[str, float],
    reference: float,
    chance: float,
    grid: Sequence[str],
    ratio: float = 0.9,
    floor_min: float = 0.3,
) -> dict:
    """ECL90 of one method on one task family (specification, section 5.4).

    curve: accuracy per grid length name; grid: the grid's length names in increasing
    order; reference: native accuracy at the reference length (8K). The result's
    "ecl" is the largest grid length L such that, at every grid length up to and
    including L, accuracy >= ratio x reference and accuracy > floor; None if the first
    grid length already fails. A grid length missing from the curve ends the scan
    (nothing beyond an unmeasured length can be certified); it is reported in "stopped_at".
    """
    floor = ecl_floor(chance, floor_min)
    threshold = ratio * reference
    ecl = None
    detail = []
    stopped = None
    for L in grid:
        if L not in curve or curve[L] is None:
            stopped = {"length": L, "reason": "not measured"}
            break
        acc = float(curve[L])
        # Tolerance for float rounding of k/n accuracies (0.9 * 0.8 is 0.7200000000000001):
        # exactly at the threshold passes, exactly at the floor does not.
        ok_ratio = acc >= threshold - ECL_EPS
        ok_floor = acc > floor + ECL_EPS
        detail.append({"length": L, "accuracy": acc, "ratio_ok": ok_ratio, "floor_ok": ok_floor})
        if not (ok_ratio and ok_floor):
            stopped = {"length": L, "reason": "below " + ("threshold" if not ok_ratio else "floor")}
            break
        ecl = L
    return {"ecl": ecl, "threshold": threshold, "floor": floor, "reference": reference, "detail": detail, "stopped_at": stopped}


# ---------------------------------------------------------------- evidence recall at budget
def _merge(spans: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted((int(s[0]), int(s[1])) for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def evidence_item_recall(evidence: Sequence[Mapping], selected: Iterable[Sequence[int]]) -> float | None:
    """Share of evidence items whose token range [tok_start, tok_end) lies wholly inside the selected ranges."""
    if not evidence:
        return None
    merged = _merge(selected)
    inside = 0
    for e in evidence:
        a, b = int(e["tok_start"]), int(e["tok_end"])
        if any(s <= a and b <= t for s, t in merged):
            inside += 1
    return inside / len(evidence)


def units_overlapping(spans: Iterable[Sequence[int]], unit_bounds: Sequence[int]) -> set[int]:
    """Indices of the units that overlap any span. unit_bounds: start token of every unit plus the end of the last."""
    bounds = np.asarray(unit_bounds)
    out: set[int] = set()
    for a, b in spans:
        if b <= a:
            continue
        first = int(np.searchsorted(bounds, a, side="right") - 1)
        last = int(np.searchsorted(bounds, b - 1, side="right") - 1)
        out.update(range(max(0, first), min(len(bounds) - 2, last) + 1))
    return out


def unit_recall_precision(gold_units: set[int], selected_units: set[int]) -> dict:
    """Evidence recall and precision over index units."""
    hit = len(gold_units & selected_units)
    return {
        "recall": hit / len(gold_units) if gold_units else None,
        "precision": hit / len(selected_units) if selected_units else None,
        "gold_units": len(gold_units),
        "selected_units": len(selected_units),
    }


# ---------------------------------------------------------------- depth deciles
def accuracy_by_decile(rows: Iterable[Mapping], n_resamples: int = 10_000, seed: int = 0) -> dict[int, dict]:
    """Accuracy and bootstrap interval per evidence depth decile (rows without a decile are left out)."""
    groups: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        d = r.get("depth_decile")
        if d is not None and r.get("score") is not None:
            groups[int(d)].append(float(r["score"]))
    return {d: bootstrap_ci(v, n_resamples, seed) for d, v in sorted(groups.items())}


# ---------------------------------------------------------------- error kinds
_SEVEN_DIGITS = re.compile(r"(?<!\d)\d{7}(?!\d)")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def error_kind(prediction: str, row: Mapping) -> str:
    """"correct", "wrong" (an answer of the expected form, but not the right one) or "format"
    (no answer of the expected form at all: empty, refusal, prose without the value).

    row needs benchmark, task ("bench:name" or the bare name), score, answers; synthetic rows
    also exclude; BABILong rows also question; InfiniteBench rows also label.
    """
    if row.get("score") == 1.0:
        return "correct"
    bench = row.get("benchmark", "synth")
    task = str(row["task"]).split(":")[-1]
    p = prediction or ""
    if bench == "synth":
        if task == "S4_count":
            ok = _INT.search(p.replace(",", "")) is not None
        elif task == "S4_freq":
            ok = any(_present(p, w) for w in list(row["answers"]) + list(row.get("exclude") or []))
        else:  # S1, S2, S3: seven-digit values
            ok = _SEVEN_DIGITS.search(p) is not None
        return "wrong" if ok else "format"
    if bench == "babilong":
        from pcm.tasks.official import babilong as off

        o = off()
        out = o.compare_answers.__globals__["preprocess_output"](p) if "preprocess_output" in o.compare_answers.__globals__ else p.lower()
        labels = {lab.lower() for lab in o.TASK_LABELS[task]}
        in_question = {lab for lab in labels if lab in str(row.get("question", "")).lower()}
        return "wrong" if {lab for lab in labels if lab in out} - in_question else "format"
    if bench == "infbench":
        if task in ("passkey", "number_string"):
            ok = re.search(r"\d", p) is not None
        elif task == "kv_retrieval":
            ok = _UUID.search(p) is not None
        else:  # longbook_choice_eng: a letter A-D or one of the options
            ok = re.search(r"\b[A-D]\b", p) is not None
        return "wrong" if ok else "format"
    raise ValueError(f"unknown benchmark {bench!r}")
