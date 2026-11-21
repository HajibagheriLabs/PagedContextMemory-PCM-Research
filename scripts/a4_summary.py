"""Numbers, tables and figures for reports/A4_baselines.md.

    python scripts/a4_summary.py

Reads results/E0_floor, E1_truncation, E2_dense, E3_retrieval and E3o_oracle (the last row per
id of each runs.jsonl; only rows with status "ok" score) and writes results/A4_baselines/
summary.json, tables.md and figures/*.png. What is compared with what is named in
configs/a4_baselines.yaml (summary). Cells that have not been run are left out, and every
table gives n, so the script can be run on a partial set of results.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.eval import metrics as M  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json, atomic_write_text  # noqa: E402
from pcm.tasks.common import config_hash, load_task_config  # noqa: E402

A4_EXPERIMENTS = ("E0_a4budget", "E2_dense", "E3_retrieval", "E3o_oracle")


def current_hashes(cfg: dict) -> dict[str, set]:
    """Method hashes of the configuration as it stands: rows of an A4 method under any other hash were run with
    settings since replaced (D-040) and are left out. M1 has one hash per YaRN factor."""
    out = {}
    for name, m in cfg["methods"].items():
        if m["method"] == "dense_yarn":
            out[name] = {config_hash({**m, "yarn_factor": f}) for f in cfg["e2"]["yarn_factors"]}
        else:
            out[name] = {config_hash(m)}
    return out

OUT = REPO_ROOT / "results" / "A4_baselines"
FIG = OUT / "figures"
ORDER = ["8k", "16k", "32k", "64k", "128k", "256k", "full"]
GIB = 1024**3


def rows_of(exp: str) -> list[dict]:
    last: dict = {}
    for r in JsonlLog(REPO_ROOT / "results" / exp / "runs.jsonl").rows():
        last[r["id"]] = r
    return list(last.values())


def chance(task: str, tcfg: dict) -> float:
    bench, name = task.split(":")
    return float(tcfg["synth"]["tasks"][name]["chance"]) if bench == "synth" else float(tcfg[bench]["chance"][name])


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


def pct(c: dict | None, digits: int = 1) -> str:
    if not c or c.get("mean") is None:
        return "–"
    return f"{100 * c['mean']:.{digits}f} [{100 * c['lo']:.{digits}f}, {100 * c['hi']:.{digits}f}]"


def family_ci(task_scores: list[list[float]], mcfg: dict) -> dict:
    """Mean of the tasks' accuracies, with a bootstrap interval that resamples inside each task."""
    if not task_scores or any(len(s) == 0 for s in task_scores):
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(mcfg["bootstrap_seed"])
    B = mcfg["bootstrap_resamples"]
    acc = np.zeros(B)
    for s in task_scores:
        x = np.asarray(s, dtype=float)
        acc += x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    acc /= len(task_scores)
    a = (1 - mcfg["ci"]) / 2
    return {"n": int(sum(len(s) for s in task_scores)), "mean": float(np.mean([np.mean(s) for s in task_scores])),
            "lo": float(np.quantile(acc, a)), "hi": float(np.quantile(acc, 1 - a))}


def main() -> int:
    tcfg = load_task_config()
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "a4_baselines.yaml").read_text(encoding="utf-8"))
    mcfg, scfg = tcfg["metrics"], cfg["summary"]
    series, families = scfg["series"], cfg["families"]
    grid = list(tcfg["lengths"])
    native = scfg["native"]

    rerun = native.get("budget_rerun") or {}
    exps = sorted({native["exp"]} | {s["exp"] for s in series.values()} | ({rerun["exp"]} if rerun else set()))
    current = current_hashes(cfg)
    rows, superseded = {}, Counter()
    for e in exps:
        keep = []
        for r in rows_of(e):
            if e in A4_EXPERIMENTS and r["method_hash"] not in current.get(r["method"], set()):
                superseded[f"{e}|{r['method']}"] += 1
                continue
            keep.append(r)
        rows[e] = keep
    commits = {e: dict(Counter(f"{(r.get('git_commit') or '')[:10]}{'+dirty' if r.get('git_dirty') else ''}" for r in rows[e])) for e in exps}
    cells: dict = defaultdict(list)  # (exp, method, task, length) -> rows
    for e in exps:
        for r in rows[e]:
            cells[(e, r["method"], r["task"], r["length"])].append(r)

    def ok_rows(exp, method, task, length):
        return [r for r in cells.get((exp, method, task, length), []) if r["status"] == "ok"]

    def ci(scores):
        return M.bootstrap_ci(scores, mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"])

    def device_gib(exp: str) -> float:
        """Memory of the GPU the experiment ran on, from its env.json."""
        env = json.loads((REPO_ROOT / "results" / exp / "env.json").read_text(encoding="utf-8"))
        return float(env["gpus"][0]["vram_total_gib"])

    # ---- every cell of the A4 experiments
    def cell_record(exp, method, task, length) -> dict:
        rs = cells[(exp, method, task, length)]
        ok = [r for r in rs if r["status"] == "ok"]
        fresh = [r for r in ok if not (r.get("timings") or {}).get("reused")]  # rows whose answer the reader produced
        sel = [r["selection"] for r in ok if r.get("selection")]
        rec = [r["evidence_recall"] for r in ok if r.get("evidence_recall") is not None]
        full = [r["score"] for r in ok if r.get("evidence_recall") == 1.0]
        part = [r["score"] for r in ok if r.get("evidence_recall") is not None and r["evidence_recall"] < 1.0]
        t = [r.get("timings") or {} for r in fresh]
        ingest = [(r["length_tokens"], r["timings"]["ingest_s"]) for r in ok if (r.get("timings") or {}).get("ingest_s")]
        out = {
            "exp": exp, "method": method, "task": task, "length": length,
            "status": dict(Counter(r["status"] for r in rs)), "accuracy": ci([r["score"] for r in ok]),
            "tokens_mean": rnd(mean([r["length_tokens"] for r in ok]), 1), "prompt_tokens_mean": rnd(mean([r["prompt_tokens"] for r in ok]), 1),
            "evidence_recall": rnd(mean(rec)), "evidence_all_kept": {"n": len(full), "accuracy": ci(full)} if rec else None,
            "evidence_not_all_kept": {"n": len(part), "accuracy": ci(part)} if rec else None,
            "reused": len(ok) - len(fresh),
            "seconds_per_example": rnd(mean([x["total_s"] - x.get("ingest_s", 0.0) for x in t]), 3),
            "first_token_s": rnd(mean([x.get("prefill_s") for x in t]), 3), "select_ms": rnd(mean([1000 * x["select_s"] for x in t if "select_s" in x]), 2),
            "rerank_ms": rnd(mean([1000 * r["timings"]["rerank_s"] for r in ok if (r.get("timings") or {}).get("rerank_s") is not None]), 2),
            "prepare_s": rnd(mean([x.get("prepare_s") for x in t]), 3),
            "peak_vram_gib": rnd(max([r.get("peak_vram_gib") or 0 for r in rs], default=None), 3),
            "peak_reserved_gib": rnd(max([r.get("peak_reserved_gib") or 0 for r in rs], default=None), 3),
            "reserved_above_device": sum(1 for r in rs if (r.get("peak_reserved_gib") or 0) > device_gib(exp)),
            "ingest": {"documents": len(ingest), "tokens_per_s": rnd(sum(a for a, _ in ingest) / sum(b for _, b in ingest), 1),
                       "seconds_per_document": rnd(mean([b for _, b in ingest]), 3)} if ingest else None,
        }
        if sel:
            out["selection"] = {k: rnd(mean([s.get(k) for s in sel]), 4) for k in ("n_units", "kept_units", "kept_tokens", "gaps", "gold_units", "unit_recall", "unit_precision")}
            out["selection"]["oracle_modes"] = dict(Counter(s["oracle"] for s in sel if "oracle" in s)) or None
            moved = [s["rerank_moved"] for s in sel if "rerank_moved" in s]
            out["selection"]["rerank_moved_mean"] = rnd(mean(moved), 2)
            ranked = [s["evidence_ranks"] for s in sel if s.get("evidence_ranks")]
            if ranked:  # evidence recall had only the top k units been kept: share of items complete within k units
                out["recall_at_units"] = {str(k): rnd(mean([sum(1 for x in rk if x <= k) / len(rk) for rk in ranked])) for k in scfg["recall_at_units"]}
        if task.startswith("babilong:"):
            out["accuracy_whole_word"] = ci([M.babilong_whole_word_score(r["prediction"], r) for r in ok])  # D-028
        budget = None
        mc_path = REPO_ROOT / "results" / exp / "methods" / f"{rs[0]['method_hash']}.json"
        if mc_path.is_file():
            mc = json.loads(mc_path.read_text(encoding="utf-8"))
            budget = (mc.get("max_new_tokens") or {}).get(task)
            if mc.get("yarn_factor") is not None:
                out["yarn_factor"] = mc["yarn_factor"]
        if budget is None and task.startswith("babilong:"):
            budget = int(tcfg["babilong"]["max_new_tokens"])
        at_budget = [r for r in ok if budget and (r.get("timings") or {}).get("new_tokens") == budget]
        out["answer_budget"] = budget
        out["answers_at_budget"] = len(at_budget) if budget else None
        out["format_at_budget"] = sum(1 for r in at_budget if M.error_kind(r["prediction"], r) == "format") if budget else None  # cut before any answer
        out["past_32768"] = sum(1 for r in ok if (r.get("prompt_tokens") or 0) + ((r.get("timings") or {}).get("new_tokens") or 0) > 32768) if exp == "E0_a4budget" else None
        out["error_kinds"] = dict(Counter(M.error_kind(r["prediction"], r) for r in ok))
        return out

    table = {}
    for (e, m, t, L) in sorted(cells, key=lambda k: (k[0], k[1], k[2], ORDER.index(k[3]) if k[3] in ORDER else 99)):
        if e in A4_EXPERIMENTS:
            table[f"{m}|{t}|{L}"] = cell_record(e, m, t, L)

    def native_source(task: str) -> tuple[str, str]:
        if rerun and task in rerun["tasks"]:
            return rerun["exp"], rerun["method"]
        return native["exp"], native["method"]

    # ---- curves: task scores per series and length (native below the series' first own length)
    def source(skey: str, task: str, length: str) -> tuple[str, str]:
        s = series[skey]
        if task.startswith("infbench:"):
            return s["exp"], s.get("infbench", s["method"])
        if grid.index(length) < grid.index(s["from"]):
            return native_source(task)
        return s["exp"], s["method"]

    def scores_by_id(skey, task, length) -> dict:
        e, m = source(skey, task, length)
        return {r["example_id"]: r["score"] for r in ok_rows(e, m, task, length)}

    curves, ecl, ratio_only = {}, {}, {}
    reference = {}
    reference_a3 = {}
    for fam, tasks in families.items():
        ref_scores = [list({r["example_id"]: r["score"] for r in ok_rows(*native_source(t), t, mcfg["reference_length"])}.values()) for t in tasks]
        reference[fam] = family_ci(ref_scores, mcfg)
        a3_scores = [list({r["example_id"]: r["score"] for r in ok_rows(native["exp"], native["method"], t, mcfg["reference_length"])}.values()) for t in tasks]
        reference_a3[fam] = family_ci(a3_scores, mcfg)  # as fixed in A3, for comparison
        ch = sum(chance(t, tcfg) for t in tasks) / len(tasks)
        for skey in series:
            curve = {}
            for L in grid:
                per_task = [list(scores_by_id(skey, t, L).values()) for t in tasks]
                if all(per_task):
                    c = family_ci(per_task, mcfg)
                    c["source"] = "native" if grid.index(L) < grid.index(series[skey]["from"]) else "method"
                    c["n_per_task"] = [len(s) for s in per_task]
                    curve[L] = c
            curves[f"{skey}|{fam}"] = curve
            if reference[fam]["mean"] is not None:
                acc = {L: c["mean"] for L, c in curve.items()}
                ecl[f"{skey}|{fam}"] = M.ecl90(acc, reference[fam]["mean"], ch, grid, mcfg["ecl_ratio"], mcfg["ecl_floor_min"])
                best = None
                for L in grid:  # the ratio condition alone (D-032): not an ECL90
                    if L not in acc or acc[L] < mcfg["ecl_ratio"] * reference[fam]["mean"] - M.ECL_EPS:
                        break
                    best = L
                ratio_only[f"{skey}|{fam}"] = best

    # ---- paired differences per family and length
    pairs = {}
    for a, b in scfg["pairs"]:
        for fam, tasks in families.items():
            for L in grid:
                if all(grid.index(L) < grid.index(series[k]["from"]) for k in (a, b)):
                    continue  # both are the native run there
                xa, xb = [], []
                for t in tasks:
                    sa, sb = scores_by_id(a, t, L), scores_by_id(b, t, L)
                    ids = sorted(set(sa) & set(sb))
                    xa += [sa[i] for i in ids]
                    xb += [sb[i] for i in ids]
                if xa:
                    pairs[f"{a}-{b}|{fam}|{L}"] = M.paired_bootstrap(xa, xb, mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"])

    # ---- family-level retrieval and oracle diagnostics
    def fam_mean(skey, fam, L, getter):
        vals = []
        for t in families[fam]:
            c = table.get(f"{series[skey]['method']}|{t}|{L}")
            v = getter(c) if c else None
            if v is None:
                return None
            vals.append(v)
        return sum(vals) / len(vals)

    recall = {}
    for skey, s in series.items():
        if s["exp"] not in ("E3_retrieval", "E3o_oracle"):
            continue
        for fam in families:
            for L in grid:
                r = fam_mean(skey, fam, L, lambda c: c["evidence_recall"])
                if r is None:
                    continue
                rec = {"evidence_recall": rnd(r), "unit_recall": rnd(fam_mean(skey, fam, L, lambda c: (c.get("selection") or {}).get("unit_recall"))),
                       "unit_precision": rnd(fam_mean(skey, fam, L, lambda c: (c.get("selection") or {}).get("unit_precision"))),
                       "kept_tokens": rnd(fam_mean(skey, fam, L, lambda c: (c.get("selection") or {}).get("kept_tokens")), 1)}
                at = {str(k): fam_mean(skey, fam, L, lambda c, k=k: (c.get("recall_at_units") or {}).get(str(k))) for k in scfg["recall_at_units"]}
                if all(v is not None for v in at.values()):
                    rec["recall_at_units"] = {k: rnd(v) for k, v in at.items()}
                # accuracy split by whether every evidence item was kept, pooled over the family's tasks
                full, part = [], []
                for t in families[fam]:
                    for r_ in ok_rows(s["exp"], s["method"], t, L):
                        if r_.get("evidence_recall") is not None:
                            (full if r_["evidence_recall"] == 1.0 else part).append(r_["score"])
                rec["evidence_all_kept"] = {"n": len(full), "accuracy": ci(full)}
                rec["evidence_not_all_kept"] = {"n": len(part), "accuracy": ci(part)}
                recall[f"{skey}|{fam}|{L}"] = rec

    # ---- cost per method and length (retained task families; InfiniteBench apart)
    cost = {}
    for skey, s in series.items():
        if s["exp"] in (native["exp"], "E1_truncation"):
            continue
        for L in grid + ["full"]:
            cs = [c for c in table.values() if c["method"] == s["method"] and c["length"] == L]
            if not cs:
                continue
            ing_docs = sum(c["ingest"]["documents"] for c in cs if c["ingest"])
            cost[f"{skey}|{L}"] = {
                "cells": len(cs), "seconds_per_example": rnd(mean([c["seconds_per_example"] for c in cs]), 3),
                "first_token_s": rnd(mean([c["first_token_s"] for c in cs]), 3), "select_ms": rnd(mean([c["select_ms"] for c in cs]), 2),
                "rerank_ms": rnd(mean([c["rerank_ms"] for c in cs]), 2), "prepare_s": rnd(mean([c["prepare_s"] for c in cs]), 3),
                "peak_vram_gib": rnd(max(c["peak_vram_gib"] or 0 for c in cs), 3), "prompt_tokens_mean": rnd(mean([c["prompt_tokens_mean"] for c in cs]), 1),
                "reused": sum(c["reused"] for c in cs), "ok": sum(c["status"].get("ok", 0) for c in cs),
                "ingest_documents": ing_docs,
                "ingest_tokens_per_s": rnd(mean([c["ingest"]["tokens_per_s"] for c in cs if c["ingest"]]), 1),
                "ingest_seconds_per_document": rnd(mean([c["ingest"]["seconds_per_document"] for c in cs if c["ingest"]]), 3),
            }

    # ---- E2: examples per cell, GPU time spent, the plan
    e2_rows = [r for r in rows.get("E2_dense", []) if r["status"] == "ok"]
    plan_path = REPO_ROOT / "results" / "E2_dense" / "n_plan.json"
    e2 = {"gpu_hours": rnd(sum(r["timings"]["total_s"] for r in e2_rows) / 3600, 3), "rows_ok": len(e2_rows),
          "plan": json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None,
          "n_per_cell": {f"{t}|{L}": c["status"].get("ok", 0) for k, c in table.items() for t, L in [(c["task"], c["length"])] if c["exp"] == "E2_dense"}}
    # Rows whose allocator held more than the card has: blocks cached from shorter examples plus a new, larger one
    # (Windows lets the driver back the excess with system memory). Their extra time, against the cell's other rows.
    if e2_rows:
        dev = device_gib("E2_dense")
        over, rest = defaultdict(list), defaultdict(list)
        for r in e2_rows:
            (over if (r.get("peak_reserved_gib") or 0) > dev else rest)[(r["task"], r["length"])].append(r["timings"]["total_s"])
        e2["reserved_above_device"] = {
            "device_gib": dev, "rows": sum(len(v) for v in over.values()), "rows_ok": len(e2_rows),
            "max_allocated_gib": rnd(max(r.get("peak_vram_gib") or 0 for r in e2_rows), 3),
            "max_reserved_gib": rnd(max(r.get("peak_reserved_gib") or 0 for r in e2_rows), 3),
            "by_cell": {f"{t}|{L}": {"rows": len(v), "of": len(v) + len(rest[(t, L)]), "s_mean": rnd(mean(v), 1), "s_mean_others": rnd(mean(rest[(t, L)]), 1)}
                        for (t, L), v in sorted(over.items())}}

    # ---- checks: results that would be bugs until explained
    checks = {"oracle_rows_missing_evidence": sum(1 for r in rows.get("E3o_oracle", []) if r["status"] == "ok" and r.get("evidence_recall") != 1.0),
              "retrieval_above_oracle": [], "above_native_reference": [], "reuse_sources_checked": 0, "reuse_mismatch": 0,
              "rerank_same_units_as_hybrid": None}
    for key, p in pairs.items():
        if key.startswith("m3-m2-hybrid|") and p["hi"] is not None and p["hi"] < 0:
            checks["retrieval_above_oracle"].append({"cell": key, "diff": p["diff"], "lo": p["lo"], "hi": p["hi"]})
    for k, curve in curves.items():
        skey, fam = k.split("|")
        ref = reference[fam]
        for L, c in curve.items():
            if c["source"] == "method" and ref["mean"] is not None and c["lo"] > ref["hi"]:
                checks["above_native_reference"].append({"series": skey, "family": fam, "length": L, "accuracy": c["mean"], "reference": ref["mean"]})
    # A reused answer may come from a pilot row whose method hash was since replaced (D-040) but whose prompt hash
    # (which covers the answer budget) is the same: look the source up among all rows of the experiment.
    e3_all = {r["id"]: r for r in rows_of("E3_retrieval")} if "E3_retrieval" in rows else {}
    kept_ids = {r["id"] for r in rows.get("E3_retrieval", [])}
    checks["reuse_from_superseded_rows"] = 0
    for r in rows.get("E3_retrieval", []):
        if r.get("reused_from"):
            src = e3_all.get(r["reused_from"])
            checks["reuse_sources_checked"] += 1
            if src is None or src.get("prompt_sha") != r.get("prompt_sha") or src.get("prediction") != r.get("prediction") or src.get("example_id") != r.get("example_id"):
                checks["reuse_mismatch"] += 1
            elif r["reused_from"] not in kept_ids:
                checks["reuse_from_superseded_rows"] += 1
    hy = {(r["task"], r["length"], r["example_id"]): r for r in rows.get("E3_retrieval", []) if r["method"] == series["m2-hybrid"]["method"] and r["status"] == "ok"}
    rr = [r for r in rows.get("E3_retrieval", []) if r["method"] == series["m2-hybrid-rerank"]["method"] and r["status"] == "ok"]
    both = [(r, hy[(r["task"], r["length"], r["example_id"])]) for r in rr if (r["task"], r["length"], r["example_id"]) in hy]
    if both:
        checks["rerank_same_units_as_hybrid"] = {"pairs": len(both), "same_units": sum(1 for a, b in both if a["selection"]["units_b64"] == b["selection"]["units_b64"]),
                                                 "same_prompt": sum(1 for a, b in both if a["prompt_sha"] == b["prompt_sha"]),
                                                 "same_score": sum(1 for a, b in both if a["score"] == b["score"])}

    # The native rerun at the A4 answer budgets against A3's native rows, example by example (D-040).
    if rerun:
        rc = {}
        for t in rerun["tasks"]:
            for L in grid:
                new = {r["example_id"]: r for r in ok_rows(rerun["exp"], rerun["method"], t, L)}
                old = {r["example_id"]: r for r in ok_rows(native["exp"], native["method"], t, L)}
                ids = sorted(set(new) & set(old))
                if ids:
                    rc[f"{t}|{L}"] = {"pairs": len(ids), "score_changed": sum(1 for i in ids if new[i]["score"] != old[i]["score"]),
                                      "same_text": sum(1 for i in ids if new[i]["prediction"] == old[i]["prediction"]),
                                      "new_extends_old": sum(1 for i in ids if new[i]["prediction"] != old[i]["prediction"] and new[i]["prediction"].startswith(old[i]["prediction"].rstrip()))}
        checks["budget_rerun_vs_a3"] = rc

    # ---- depth deciles of the synthetic tasks at the longest length each series has
    deciles = {}
    for skey, s in series.items():
        for t in [t for ts in families.values() for t in ts if t.startswith("synth:")]:
            for L in reversed(grid):
                ok = ok_rows(s["exp"], s["method"], t, L)
                if ok and grid.index(L) >= grid.index(s["from"]):
                    by = M.accuracy_by_decile(ok, mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"])
                    if by:
                        deciles[f"{skey}|{t}|{L}"] = {str(d): {"n": v["n"], "mean": v["mean"]} for d, v in by.items()}
                    break

    # ---- InfiniteBench (one length, no native reference)
    infbench = {}
    for skey, s in series.items():
        for t in cfg["task_sets"]["infbench"]:
            e, m = source(skey, t, "full")
            ok = ok_rows(e, m, t, "full")
            if ok:
                rec = [r["evidence_recall"] for r in ok if r.get("evidence_recall") is not None]
                infbench[f"{skey}|{t}"] = {"method": m, "accuracy": ci([r["score"] for r in ok]), "evidence_recall": rnd(mean(rec)),
                                           "tokens_mean": rnd(mean([r["length_tokens"] for r in ok]), 1)}

    # ---- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    figs = []
    styles = {"m0-tail": ("tab:gray", "^--"), "m1": ("tab:red", "s-"), "m2-bm25": ("tab:olive", "v:"), "m2-embed": ("tab:cyan", "<:"),
              "m2-hybrid": ("tab:blue", "o-"), "m3": ("tab:green", "D-"), "m3-tight": ("tab:green", "d--")}
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4.2), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), families):
        for skey, (color, style) in styles.items():
            curve = curves.get(f"{skey}|{fam}", {})
            Ls = [L for L in grid if L in curve]
            if not Ls:
                continue
            y = [100 * curve[L]["mean"] for L in Ls]
            err = [[100 * (curve[L]["mean"] - curve[L]["lo"]) for L in Ls], [100 * (curve[L]["hi"] - curve[L]["mean"]) for L in Ls]]
            ax.errorbar([grid.index(L) for L in Ls], y, yerr=err, fmt=style, color=color, capsize=2, markersize=4, linewidth=1.2, label=series[skey]["label"])
        if reference[fam]["mean"] is not None:
            ax.axhline(100 * mcfg["ecl_ratio"] * reference[fam]["mean"], color="black", linewidth=0.8, linestyle="-.")
        ax.set_title(fam, fontsize=10)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels(grid, fontsize=8)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("context length")
    np.atleast_1d(axes)[0].set_ylabel("accuracy (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Accuracy vs length, Qwen3-1.7B (95% CI). Dash-dot line: 90% of the native 8K reference. Below its first own length a curve is the native run.", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "curves.png", dpi=120)
    plt.close(fig)
    figs.append("figures/curves.png")

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 3.6), sharey=True)
    drew = False
    for ax, fam in zip(np.atleast_1d(axes), families):
        for skey in ("m2-bm25", "m2-embed", "m2-hybrid"):
            Ls = [L for L in grid if f"{skey}|{fam}|{L}" in recall]
            if Ls:
                drew = True
                ax.plot([grid.index(L) for L in Ls], [recall[f"{skey}|{fam}|{L}"]["evidence_recall"] for L in Ls], styles[skey][1], color=styles[skey][0], label=series[skey]["label"])
        ax.set_title(fam, fontsize=10)
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels(grid, fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.9, color="black", linewidth=0.6, linestyle=":")
    np.atleast_1d(axes)[0].set_ylabel("evidence recall at budget")
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("M2: share of evidence items inside the 28,672 kept tokens", fontsize=9)
    fig.tight_layout()
    if drew:
        fig.savefig(FIG / "recall.png", dpi=120)
        figs.append("figures/recall.png")
    plt.close(fig)

    summary = {"commits": commits, "superseded_rows": dict(superseded), "reference": reference, "reference_a3": reference_a3, "cells": table, "curves": curves, "ecl90": ecl, "ratio_only": ratio_only,
               "pairs": pairs, "retrieval": recall, "cost": cost, "e2": e2, "checks": checks, "deciles": deciles, "infbench": infbench, "figures": figs}
    atomic_write_json(OUT / "summary.json", summary)

    # ---- tables
    lab = {k: v["label"] for k, v in series.items()}
    out = ["# A4 tables (written by scripts/a4_summary.py from the results files)", "", f"Code commits of the rows: {json.dumps(commits)}", "",
           f"Rows left out, run with settings since replaced (D-040): {json.dumps(dict(superseded))}", "",
           "References (native 8K, mean of the family's tasks): " + "; ".join(
               f"{fam} {pct(reference[fam])} (A3: {pct(reference_a3[fam])})" for fam in families), ""]
    for fam in families:
        ref = reference[fam]
        out += [f"## {fam}: accuracy (%, 95% CI), n per task in brackets. Reference (native 8K): {pct(ref)}; 90% of it: {100 * mcfg['ecl_ratio'] * ref['mean']:.2f}" if ref["mean"] is not None else f"## {fam}", "",
                "| Method | " + " | ".join(grid) + " | ECL90 | stops at |", "|---|" + "---|" * (len(grid) + 2)]
        for skey in series:
            curve = curves.get(f"{skey}|{fam}", {})
            if not curve:
                continue
            e = ecl.get(f"{skey}|{fam}", {})
            st = e.get("stopped_at")
            cells_txt = [(pct(curve[L]) + (" (native)" if curve[L]["source"] == "native" else f" (n={min(curve[L]['n_per_task'])})")) if L in curve else "–" for L in grid]
            out.append(f"| {lab[skey]} | " + " | ".join(cells_txt) + f" | {e.get('ecl') or 'none'} | {(st['length'] + ': ' + st['reason']) if st else 'end of grid'} |")
        out.append("")
    out += ["## ECL90 by family", "", "| Method | " + " | ".join(families) + " |", "|---|" + "---|" * len(families)]
    for skey in series:
        out.append(f"| {lab[skey]} | " + " | ".join(str((ecl.get(f'{skey}|{fam}') or {}).get('ecl') or 'none') for fam in families) + " |")
    out += ["", "## Ratio condition alone (accuracy >= 90% of the reference at every length up to L); not an ECL90 (D-032)", "",
            "| Method | " + " | ".join(families) + " |", "|---|" + "---|" * len(families)]
    for skey in series:
        out.append(f"| {lab[skey]} | " + " | ".join(str(ratio_only.get(f"{skey}|{fam}") or "none") for fam in families) + " |")
    out += ["", "## Paired differences (points, 95% CI; first minus second)", "", "| Pair | Family | Length | n | Difference | Excludes 0 |", "|---|---|---|---|---|---|"]
    for key, p in pairs.items():
        pair, fam, L = key.split("|")
        a, b = next((x, y) for x, y in scfg["pairs"] if f"{x}-{y}" == pair)
        out.append(f"| {lab[a]} minus {lab[b]} | {fam} | {L} | {p['n']} | {100 * p['diff']:+.1f} [{100 * p['lo']:+.1f}, {100 * p['hi']:+.1f}] | {'yes' if p['excludes_0'] else 'no'} |")
    out += ["", "## Evidence at budget (family means over tasks)", "",
            "| Method | Family | Length | Item recall | Unit recall | Unit precision | Kept tokens | Accuracy, all evidence kept (n) | Accuracy, some evidence missing (n) | Recall at "
            + ", ".join(str(k) for k in scfg["recall_at_units"]) + " units |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in recall.items():
        skey, fam, L = key.split("|")
        at = r.get("recall_at_units")
        out.append(f"| {lab[skey]} | {fam} | {L} | {r['evidence_recall']} | {r['unit_recall']} | {r['unit_precision']} | {r['kept_tokens']} | "
                   f"{pct(r['evidence_all_kept']['accuracy'])} ({r['evidence_all_kept']['n']}) | {pct(r['evidence_not_all_kept']['accuracy'])} ({r['evidence_not_all_kept']['n']}) | "
                   + (" / ".join(f"{at[str(k)]:.3f}" for k in scfg["recall_at_units"]) if at else "–") + " |")
    out += ["", "## Cost (means over the cells of a method at a length)", "",
            "| Method | Length | Cells | Rows ok | Reused answers | Prompt tokens | s per example | First token s | Selection ms | Rerank ms | CPU preparation s | Peak VRAM GiB | Ingest: documents, tokens/s, s per document |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in cost.items():
        skey, L = key.split("|")
        out.append(f"| {lab[skey]} | {L} | {c['cells']} | {c['ok']} | {c['reused']} | {c['prompt_tokens_mean']} | {c['seconds_per_example']} | {c['first_token_s']} | {c['select_ms']} | {c['rerank_ms']} | "
                   f"{c['prepare_s']} | {c['peak_vram_gib']} | {c['ingest_documents']}, {c['ingest_tokens_per_s']}, {c['ingest_seconds_per_document']} |")
    out += ["", "## InfiniteBench (accuracy %, 95% CI; evidence recall)", "", "| Method | " + " | ".join(t.split(":")[1] for t in cfg["task_sets"]["infbench"]) + " |",
            "|---|" + "---|" * len(cfg["task_sets"]["infbench"])]
    for skey in series:
        if any(f"{skey}|{t}" in infbench for t in cfg["task_sets"]["infbench"]):
            out.append(f"| {lab[skey]} | " + " | ".join(
                (pct(infbench[f"{skey}|{t}"]["accuracy"]) + f" (n={infbench[f'{skey}|{t}']['accuracy']['n']}"
                 + (f", recall {infbench[f'{skey}|{t}']['evidence_recall']}" if infbench[f"{skey}|{t}"]["evidence_recall"] is not None else "") + ")")
                if f"{skey}|{t}" in infbench else "–" for t in cfg["task_sets"]["infbench"]) + " |")
    out += ["", "## Every cell of E2, E3 and E3o", "",
            "| Method | Task | Length | Status | Accuracy | Whole-word (BABILong) | Evidence recall | s per example | Peak VRAM GiB | Error kinds |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in table.values():
        out.append(f"| {c['method']} | {c['task']} | {c['length']} | {c['status']} | {pct(c['accuracy'])} | {pct(c.get('accuracy_whole_word'))} | {c['evidence_recall']} | "
                   f"{c['seconds_per_example']} | {c['peak_vram_gib']} | {c['error_kinds']} |")
    out += ["", "## E2", "", json.dumps(e2), "", "## Checks", "", json.dumps(checks), ""]
    atomic_write_text(OUT / "tables.md", "\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
