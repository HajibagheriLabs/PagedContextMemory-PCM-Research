"""Numbers, tables and figures for reports/A5_pillars12.md.

    python scripts/a5_summary.py

Reads results/E4_pages (screen and final stages) and results/E5_graph. For comparison: A4's M2 hybrid and M3
oracle rows at A4's current method hashes (results/E3_retrieval, results/E3o_oracle). For ingest cost at 256K:
the embedding log of the index cache (data/cache/index/.../ingest.jsonl), the graph build times recorded in E5
rows, the spaCy entity log (results/E5_graph/ingest_entities_spacy.jsonl) and the sizes of the cached files.
Writes results/A5_pillars12/summary.json, tables.md and figures/*.png. Rules and task groups come from
configs/a5_pillars.yaml; how H2 is read is D-045. Every table gives n.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcm.eval import metrics as M  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json, atomic_write_text  # noqa: E402
from pcm.tasks.common import config_hash, load_task_config  # noqa: E402
from run_a5 import cells_for, load_config, screen_settings, setting_config  # noqa: E402

OUT = REPO_ROOT / "results" / "A5_pillars12"
FIG = OUT / "figures"
MIB = 1024**2
EPS = 1e-9


def rows_of(exp: str) -> list[dict]:
    p = REPO_ROOT / "results" / exp / "runs.jsonl"
    if not p.is_file():
        return []
    last: dict = {}
    for r in JsonlLog(p).rows():
        last[r["id"]] = r
    return list(last.values())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


def pct(c, digits=1):
    if not c or c.get("mean") is None:
        return "–"
    return f"{100 * c['mean']:.{digits}f} [{100 * c['lo']:.{digits}f}, {100 * c['hi']:.{digits}f}]"


def dpct(d, digits=1):
    if not d or d.get("diff") is None:
        return "–"
    return f"{100 * d['diff']:+.{digits}f} [{100 * d['lo']:+.{digits}f}, {100 * d['hi']:+.{digits}f}]"


def f2(x, k=2):
    return "–" if x is None else f"{x:.{k}f}"


def family_ci(task_scores, B, seed, ci):
    """Mean of the tasks' accuracies, with a bootstrap interval that resamples inside each task (as in A4)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(B)
    for s in task_scores:
        x = np.asarray(s, dtype=float)
        acc += x[rng.integers(0, x.size, size=(B, x.size))].mean(axis=1)
    acc /= len(task_scores)
    a = (1 - ci) / 2
    return {"n": int(sum(len(s) for s in task_scores)), "mean": float(np.mean([np.mean(s) for s in task_scores])),
            "lo": float(np.quantile(acc, a)), "hi": float(np.quantile(acc, 1 - a))}


def sel(r, k):
    return (r.get("selection") or {}).get(k)


def main() -> int:
    tcfg = load_task_config()
    cfg = load_config()
    mcfg = tcfg["metrics"]
    B, SEED, CI = mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"]

    def ci(xs):
        return M.bootstrap_ci(list(xs), B, SEED, CI)

    def paired(a, b):
        return M.paired_bootstrap(a, b, B, SEED, CI)

    exps = {"E4_pages": rows_of("E4_pages"), "E5_graph": rows_of("E5_graph")}
    commits = {e: dict(Counter(f"{(r.get('git_commit') or '')[:10]}{'+dirty' if r.get('git_dirty') else ''}" for r in rs)) for e, rs in exps.items()}
    status = {e: dict(Counter(r["status"] for r in rs)) for e, rs in exps.items()}
    prompt_max = {e: max([r.get("prompt_tokens") or 0 for r in rs] or [0]) for e, rs in exps.items()}

    cells = defaultdict(list)  # (exp, method_hash, task, length) -> ok rows
    for e, rs in exps.items():
        for r in rs:
            if r["status"] == "ok":
                cells[(e, r["method_hash"], r["task"], r["length"])].append(r)

    def cell(exp, mc, task, L):
        return cells.get((exp, config_hash(mc), task, L), [])

    def scell(mc, task, L):
        """A screen cell: its first n examples only (the final stage added rows to the three best settings)."""
        return [r for r in cells.get(("E4_pages", config_hash(mc), task, L), []) if r["index"] < cfg["e4"]["screen"]["n"]]

    def acc_of(rs):
        return [r["score"] for r in rs if r.get("score") is not None]

    def recall_of(rs):
        return mean([r.get("evidence_recall") for r in rs])

    def pair_rows(ra, rb, value="score"):
        a = [r for r in ra if r.get(value) is not None]
        b = [r for r in rb if r.get(value) is not None]
        return M.paired_by_id(a, b, value=value)

    # ---------------------------------------------------------------- E4 screen: the grid
    sc = cfg["e4"]["screen"]
    pages, budgets = cfg["e4"]["pages"], cfg["e4"]["budgets"]
    grid = {}
    for p, b in screen_settings(cfg):
        mc = setting_config(cfg, p, b)
        rec = {"page": p, "budget": b, "cells": {}}
        for L in sc["lengths"]:
            for t in sc["tasks"]:
                rs = scell(mc, t, L)
                rec["cells"][f"{t}|{L}"] = {
                    "n": len(rs), "accuracy": ci(acc_of(rs)) if rs else None, "evidence_recall": rnd(recall_of(rs)),
                    "unit_recall": rnd(mean([sel(r, "unit_recall") for r in rs])), "unit_precision": rnd(mean([sel(r, "unit_precision") for r in rs])),
                    "kept_tokens": rnd(mean([sel(r, "kept_tokens") for r in rs]), 1), "kept_pages": rnd(mean([sel(r, "kept_pages") for r in rs]), 2),
                    "gaps": rnd(mean([sel(r, "gaps") for r in rs]), 2), "prompt_tokens": rnd(mean([r.get("prompt_tokens") for r in rs]), 1),
                    "seconds_per_example": rnd(mean([(r.get("timings") or {}).get("total_s") for r in rs]), 3)}
            for t in sc["recall_only"]:
                rs = scell(setting_config(cfg, p, b, selection_only=True), t, L)
                rec["cells"][f"{t}|{L}"] = {"n": len(rs), "evidence_recall": rnd(recall_of(rs)), "unit_recall": rnd(mean([sel(r, "unit_recall") for r in rs]))}
        acc_cells = [rec["cells"][f"{t}|{L}"] for L in sc["lengths"] for t in sc["tasks"]]
        if all(c["accuracy"] for c in acc_cells):
            rec["accuracy_mean"] = rnd(mean([c["accuracy"]["mean"] for c in acc_cells]))
            rec["recall_mean"] = rnd(mean([c["evidence_recall"] for c in acc_cells]))
            rec["unit_precision_mean"] = rnd(mean([c["unit_precision"] for c in acc_cells]))
            rec["gaps_mean"] = rnd(mean([c["gaps"] for c in acc_cells]), 2)
            rec["kept_tokens_mean"] = rnd(mean([c["kept_tokens"] for c in acc_cells]), 1)
            for L in sc["lengths"]:
                rec[f"accuracy_{L}"] = rnd(mean([rec["cells"][f"{t}|{L}"]["accuracy"]["mean"] for t in sc["tasks"]]))
                rec[f"recall_{L}"] = rnd(mean([rec["cells"][f"{t}|{L}"]["evidence_recall"] for t in sc["tasks"]]))
            for t in sc["tasks"]:
                rec[f"accuracy_{t}"] = rnd(mean([rec["cells"][f"{t}|{L}"]["accuracy"]["mean"] for L in sc["lengths"]]))
                rec[f"recall_{t}"] = rnd(mean([rec["cells"][f"{t}|{L}"]["evidence_recall"] for L in sc["lengths"]]))
        rec["s3_recall_mean"] = rnd(mean([rec["cells"][f"{t}|{L}"]["evidence_recall"] for L in sc["lengths"] for t in sc["recall_only"]]))
        grid[f"p{p}|b{b}"] = rec
    screen_rows = sum(c["n"] for g in grid.values() for c in g["cells"].values())

    # Paired: at each budget, every page size against page 512 (pooled over the screen's accuracy cells).
    page_pairs = {}
    for b in budgets:
        if 512 > b:
            continue
        ref = setting_config(cfg, 512, b)
        for p in pages:
            if p > b or p == 512:
                continue
            mc = setting_config(cfg, p, b)
            a, r_ = [], []
            for L in sc["lengths"]:
                for t in sc["tasks"]:
                    x, y, _ = pair_rows(scell(mc, t, L), scell(ref, t, L))
                    a += x
                    r_ += y
            page_pairs[f"p{p}-p512|b{b}"] = paired(a, r_)

    best_path = REPO_ROOT / "results" / "E4_pages" / "best_settings.json"
    e5_path = REPO_ROOT / "results" / "E4_pages" / "e5_setting.json"
    best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.is_file() else None
    e5_setting = json.loads(e5_path.read_text(encoding="utf-8")) if e5_path.is_file() else None

    # ---------------------------------------------------------------- A4 comparison rows (M2 hybrid at 28,672, M3 oracles)
    a4cfg = yaml.safe_load((REPO_ROOT / "configs" / "a4_baselines.yaml").read_text(encoding="utf-8"))
    a4 = {}
    for key in ("m2_hybrid", "m3_oracle", "m3_tight"):
        src = cfg["summary"][key]
        h = config_hash(a4cfg["methods"][src["method"]])
        by = defaultdict(list)
        for r in rows_of(src["exp"]):
            if r["status"] == "ok" and r["method_hash"] == h:
                by[(r["task"], r["length"])].append(r)
        a4[key] = by

    # ---------------------------------------------------------------- E4 final: the three settings on every task
    fams = cfg["families"]
    flens = cfg["e4"]["final"]["lengths"]
    final, final_pairs, a4_fams = {}, {}, {}
    e5_mc = setting_config(cfg, e5_setting["page"], e5_setting["budget"]) if e5_setting else None
    if best:
        for s in best["settings"]:
            mc = setting_config(cfg, s["page"], s["budget"])
            key = f"p{s['page']}|b{s['budget']}"
            rec = {"tasks": {}, "families": {}}
            for f, tasks in fams.items():
                for L in flens:
                    per = []
                    for t in tasks:
                        rs = cell("E4_pages", mc, t, L)
                        rec["tasks"][f"{t}|{L}"] = {"n": len(rs), "accuracy": ci(acc_of(rs)) if rs else None, "evidence_recall": rnd(recall_of(rs)),
                                                    "seconds_per_example": rnd(mean([(r.get("timings") or {}).get("total_s") for r in rs]), 3),
                                                    "prompt_tokens": rnd(mean([r.get("prompt_tokens") for r in rs]), 1)}
                        per.append(acc_of(rs))
                    if all(per):
                        rec["families"][f"{f}|{L}"] = family_ci(per, B, SEED, CI)
                        rec["families"][f"{f}|{L}"]["evidence_recall"] = rnd(mean([rec["tasks"][f"{t}|{L}"]["evidence_recall"] for t in tasks]))
                    if e5_mc is not None and config_hash(mc) != config_hash(e5_mc):
                        a, b_ = [], []
                        for t in tasks:
                            x, y, _ = pair_rows(cell("E4_pages", mc, t, L), cell("E4_pages", e5_mc, t, L))
                            a += x
                            b_ += y
                        final_pairs[f"{key.replace('|', '-')}-vs-e5|{f}|{L}"] = paired(a, b_)
            for t in cfg["task_sets"]["infbench"]:
                rs = cell("E4_pages", mc, t, "full")
                rec["tasks"][f"{t}|full"] = {"n": len(rs), "accuracy": ci(acc_of(rs)) if rs else None, "evidence_recall": rnd(recall_of(rs))}
            rec["cost"] = {}
            for L in flens:
                rs = [r for t in [t for ts in fams.values() for t in ts] for r in cell("E4_pages", mc, t, L)]
                rec["cost"][L] = {"rows": len(rs), "seconds_per_example": rnd(mean([(r.get("timings") or {}).get("total_s") for r in rs]), 3),
                                  "prompt_tokens": rnd(mean([r.get("prompt_tokens") for r in rs]), 1),
                                  "peak_vram_gib": rnd(max([r.get("peak_vram_gib") or 0 for r in rs] or [0]), 2)}
            final[key] = rec
        for key, by in a4.items():
            for f, tasks in fams.items():
                for L in flens:
                    per = [acc_of(by.get((t, L), [])) for t in tasks]
                    if all(per):
                        a4_fams[f"{key}|{f}|{L}"] = family_ci(per, B, SEED, CI)
                        a4_fams[f"{key}|{f}|{L}"]["evidence_recall"] = rnd(mean([recall_of(by.get((t, L), [])) for t in tasks]))
                    # paired against the E5 setting (same example ids)
                    if e5_mc is not None:
                        a, b_ = [], []
                        for t in tasks:
                            x, y, _ = pair_rows(cell("E4_pages", e5_mc, t, L), by.get((t, L), []))
                            a += x
                            b_ += y
                        if a:
                            final_pairs[f"e5-vs-{key}|{f}|{L}"] = paired(a, b_)
            for t in cfg["task_sets"]["infbench"]:
                rs = by.get((t, "full"), [])
                if rs:
                    a4_fams[f"{key}|{t}|full"] = {**ci(acc_of(rs)), "evidence_recall": rnd(recall_of(rs))}

    # ---------------------------------------------------------------- E5: graph ablation
    e5 = {"variants": {}, "h2": {}, "lengths": {}}
    h2cfg = cfg["h2"]
    ran = defaultdict(set)  # method name -> {(task, length)}
    if e5_setting:
        p, b = e5_setting["page"], e5_setting["budget"]
        plan = cells_for(cfg, "e5", e5_setting)
        for c in plan:
            ran[c["method"]].add((c["task"], c["length"]))
        variants = list(cfg["e5"]["variants"])

        def vcfg(v, so=False):
            return setting_config(cfg, p, b, v, selection_only=so)

        def vrows(v, t, L, so=False):
            return cell("E5_graph", vcfg(v, so), t, L)

        for v in variants:
            for so in (False, True):
                mc = vcfg(v, so)
                if mc["name"] not in ran:
                    continue
                rec = {}
                for t, L in sorted(ran[mc["name"]]):
                    rs = vrows(v, t, L, so)
                    base = vrows("none", t, L, so)
                    d = {"n": len(rs), "evidence_recall": rnd(recall_of(rs)), "unit_recall": rnd(mean([sel(r, "unit_recall") for r in rs]))}
                    if not so:
                        d["accuracy"] = ci(acc_of(rs)) if rs else None
                        d["seconds_per_example"] = rnd(mean([(r.get("timings") or {}).get("total_s") for r in rs]), 3)
                    d["graph_s"] = rnd(mean([(r.get("timings") or {}).get("graph_s") for r in rs]), 4)
                    d["moved_in_top_seeds"] = rnd(mean([(sel(r, "graph") or {}).get("moved_in_top_seeds") for r in rs]), 2)
                    qe = [(sel(r, "graph") or {}).get("question_entities") for r in rs]
                    qe = [x for x in qe if x is not None]
                    if qe:
                        d["question_entities_mean"] = rnd(mean(qe), 2)
                        d["question_entities_zero_share"] = rnd(mean([1.0 if x == 0 else 0.0 for x in qe]), 3)
                    if v != "none":
                        if not so:
                            x, y, _ = pair_rows(rs, base)
                            d["diff_vs_none"] = paired(x, y)
                        x, y, _ = pair_rows(rs, base, "evidence_recall")
                        d["recall_diff_vs_none"] = paired(x, y)
                        d["same_kept_units"] = int(sum(1 for r in rs for q in base if q["example_id"] == r["example_id"] and sel(r, "units_b64") == sel(q, "units_b64")))
                    rec[f"{t}|{L}"] = d
                e5["variants"][mc["name"]] = rec

        def pooled(v, tasks, L):
            a, b_, ra, rb = [], [], [], []
            for t in tasks:
                if (t, L) not in ran[vcfg(v)["name"]]:
                    continue
                x, y, _ = pair_rows(vrows(v, t, L), vrows("none", t, L))
                a += x
                b_ += y
                x, y, _ = pair_rows(vrows(v, t, L), vrows("none", t, L), "evidence_recall")
                ra += x
                rb += y
            out = paired(a, b_)
            out["tasks"] = [t for t in tasks if (t, L) in ran[vcfg(v)["name"]]]
            out["variant_mean"] = rnd(mean(a))
            out["none_mean"] = rnd(mean(b_))
            out["recall_diff"] = rnd(mean(ra) - mean(rb)) if ra else None
            return out

        single = [t for t in h2cfg["single_fact"] if not t.startswith("infbench:")]
        for v in variants:
            if v == "none":
                continue
            rec = {"lengths": {}}
            holds = True
            for L in h2cfg["lengths"]:
                multi = pooled(v, h2cfg["multi_fact_primary"], L)
                sing = pooled(v, single, L)
                sec = pooled(v, h2cfg["multi_fact_secondary"], L)
                m_pass = multi["n"] > 0 and multi["diff"] >= h2cfg["gain_points"] / 100 - EPS and multi["lo"] > 0
                s_pass = sing["n"] > 0 and abs(sing["diff"]) <= h2cfg["single_band"] / 100 + EPS
                rec["lengths"][L] = {"multi_fact": multi, "single_fact": sing, "secondary_multi_fact": sec, "multi_pass": bool(m_pass), "single_pass": bool(s_pass)}
                holds = holds and m_pass and s_pass
            rec["holds"] = bool(holds)
            rec["extractor"] = (cfg["e5"]["variants"][v] or {}).get("extractor", "default")
            e5["h2"][v] = rec
        default_holds = [v for v, r in e5["h2"].items() if r["holds"] and r["extractor"] == "default"]
        e5["h2_supported"] = bool(default_holds)
        e5["h2_holding_variants"] = [v for v, r in e5["h2"].items() if r["holds"]]

        # Across lengths: no graph and the full variant of each traversal, by family.
        for v in ("none", "ppr-all", "hops2"):
            for f in ("S1", "S2", "BABILong", "BABILong-ext"):
                for L in ("32k", "64k", "128k", "256k"):
                    per = [acc_of(vrows(v, t, L)) for t in fams[f]]
                    if all(per):
                        rec = family_ci(per, B, SEED, CI)
                        rec["evidence_recall"] = rnd(mean([recall_of(vrows(v, t, L)) for t in fams[f]]))
                        if v != "none":
                            a, b_ = [], []
                            for t in fams[f]:
                                x, y, _ = pair_rows(vrows(v, t, L), vrows("none", t, L))
                                a += x
                                b_ += y
                            rec["diff_vs_none"] = paired(a, b_)
                        e5["lengths"][f"{v}|{f}|{L}"] = rec

    # ---------------------------------------------------------------- Gate G1
    g1c = cfg["g1"]
    g1 = {"threshold": g1c["threshold"], "length": g1c["length"], "settings": {}}

    def g1_of(rows_by_task):
        rec = {}
        for t in g1c["tasks"] + g1c["beside"]:
            rs = rows_by_task(t)
            vals = [r.get("evidence_recall") for r in rs if r.get("evidence_recall") is not None]
            rec[t] = {"n": len(vals), "evidence_recall": rnd(mean(vals)), "share_full": rnd(mean([1.0 if v >= 1 - EPS else 0.0 for v in vals])),
                      "accuracy": rnd(mean(acc_of(rs)))}
        rec["pass"] = all(rec[t]["evidence_recall"] is not None and rec[t]["evidence_recall"] >= g1c["threshold"] for t in g1c["tasks"])
        return rec

    if best:
        for s in best["settings"]:
            mc = setting_config(cfg, s["page"], s["budget"])
            g1["settings"][mc["name"]] = g1_of(lambda t, mc=mc: cell("E4_pages", mc, t, "full" if t.startswith("infbench:") else g1c["length"]))
    if e5_setting:
        for v in cfg["e5"]["variants"]:
            mc = setting_config(cfg, e5_setting["page"], e5_setting["budget"], v)
            if (g1c["tasks"][0], g1c["length"]) in ran.get(mc["name"], set()):
                g1["settings"][mc["name"] + " (E5)"] = g1_of(lambda t, mc=mc: cell("E5_graph", mc, t, "full" if t.startswith("infbench:") else g1c["length"]))

    # ---------------------------------------------------------------- ingest time and index size at 256K
    ingest = {}
    idx_dirs = [d for d in (REPO_ROOT / "data" / "cache" / "index").glob("*/*/u128s32") if (d / "ingest.jsonl").is_file()]
    if len(idx_dirs) == 1:
        idir = idx_dirs[0]
        log = [json.loads(line) for line in (idir / "ingest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        log = [r for r in log if "-256k-" in str(r.get("example", ""))]
        groups = defaultdict(list)
        for r in log:
            groups["synthetic" if not str(r["example"]).startswith("babilong") else "babilong"].append(r)
        for g, rs in groups.items():
            sizes = defaultdict(list)
            for r in rs:
                doc = r["doc"]
                files = {"embeddings": idir / doc[:2] / f"{doc}.npz",
                         "sem_edges": idir / "graph" / "sem-k5-m0.5" / doc[:2] / f"{doc}.npz",
                         "entities_default": idir / "graph" / "ent-default-c20" / doc[:2] / f"{doc}.npz",
                         "entities_spacy": idir / "graph" / "ent-spacy-c20" / doc[:2] / f"{doc}.npz"}
                for k, f in files.items():
                    if f.is_file():
                        sizes[k].append(f.stat().st_size)
            ingest[g] = {"docs": len(rs), "tokens": rnd(mean([r["tokens"] for r in rs]), 0), "units": rnd(mean([r["units"] for r in rs]), 1),
                         "embed_s": {"mean": rnd(mean([r["embed_s"] for r in rs]), 3), "min": rnd(min(r["embed_s"] for r in rs), 3), "max": rnd(max(r["embed_s"] for r in rs), 3)},
                         "tokens_per_s": rnd(mean([r["tokens_per_s"] for r in rs]), 0),
                         "bytes": {k: {"n": len(v), "mean_mib": rnd(np.mean(v) / MIB, 3), "bytes_per_token": rnd(np.mean(v) / mean([r["tokens"] for r in rs]), 2)} for k, v in sizes.items()}}
        ingest["index_dir"] = str(idir.relative_to(REPO_ROOT)).replace("\\", "/")
    builds = defaultdict(list)
    for r in exps["E5_graph"]:
        if r["length"] != "256k" or r["status"] != "ok":
            continue
        g = "babilong" if r["task"].startswith("babilong") else "synthetic"
        for k in ("sem_build_s", "ent_build_s"):
            v = (r.get("timings") or {}).get(k)
            if v is not None:
                builds[f"{g}|{k}"].append(v)
    for k, v in builds.items():
        g, name = k.split("|")
        ingest.setdefault(g, {})[name] = {"n": len(v), "mean": rnd(mean(v), 3), "max": rnd(max(v), 3)}
    sp = REPO_ROOT / "results" / "E5_graph" / "ingest_entities_spacy.jsonl"
    if sp.is_file():
        rs = [json.loads(line) for line in sp.read_text(encoding="utf-8").splitlines() if line.strip()]
        rs = [r for r in rs if r.get("length") == "256k" and not r.get("cached")]
        if rs:
            ingest.setdefault("babilong", {})["spacy_entities_s"] = {"n": len(rs), "mean": rnd(mean([r["s"] for r in rs]), 2), "max": rnd(max(r["s"] for r in rs), 2)}
    # Query-time selection cost at 256K (BM25 built from the units at query time, the query embedding, RRF, graph, pages).
    qcost = {}
    if e5_setting:
        for v in cfg["e5"]["variants"]:
            mc = setting_config(cfg, e5_setting["page"], e5_setting["budget"], v)
            rs = [r for (e, h, t, L), rr in cells.items() if e == "E5_graph" and h == config_hash(mc) and L == "256k" for r in rr]
            if rs:
                qcost[v] = {"n": len(rs), "select_s": rnd(mean([(r.get("timings") or {}).get("select_s") for r in rs]), 3),
                            "graph_s": rnd(mean([(r.get("timings") or {}).get("graph_s") for r in rs]), 4),
                            "prepare_s": rnd(mean([(r.get("timings") or {}).get("prepare_s") for r in rs]), 3)}
    ingest["query_time_256k"] = qcost

    # ---------------------------------------------------------------- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    figs = []

    def mat(fn):
        m = np.full((len(pages), len(budgets)), np.nan)
        for i, p in enumerate(pages):
            for j, b in enumerate(budgets):
                g = grid.get(f"p{p}|b{b}")
                if g is not None and fn(g) is not None:
                    m[i, j] = fn(g)
        return m

    def heat(ax, m, title, scale=100, vmin=0, vmax=100, fmt="{:.0f}"):
        im = ax.imshow(m * scale, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if not np.isnan(m[i, j]):
                    v = m[i, j] * scale
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="white" if v < (vmin + vmax) / 2 else "black")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b // 1024}K" for b in budgets], fontsize=8)
        ax.set_yticks(range(len(pages)))
        ax.set_yticklabels(pages, fontsize=8)
        ax.set_xlabel("budget (tokens)", fontsize=8)
        ax.set_title(title, fontsize=9)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for row, L in enumerate(sc["lengths"]):
        heat(axes[row, 0], mat(lambda g, L=L: g.get(f"accuracy_{L}")), f"accuracy (%), {L}: S2 multi-key, qa2, qa3", vmin=20, vmax=60)
        heat(axes[row, 1], mat(lambda g, L=L: g.get(f"recall_{L}")), f"evidence recall (%), {L}: same cells", vmin=0, vmax=100)
        heat(axes[row, 2], mat(lambda g, L=L: mean([g["cells"][f"{t}|{L}"]["evidence_recall"] for t in sc["recall_only"]])), f"evidence recall (%), {L}: S3 chain (h3)", vmin=0, vmax=100)
        axes[row, 0].set_ylabel("page size (tokens)", fontsize=8)
    fig.suptitle("E4 screen, N = 30 per cell: PCM text mode, hybrid ranking, no graph. Blank: page larger than the budget.", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e4_heatmaps.png", dpi=120)
    plt.close(fig)
    figs.append("figures/e4_heatmaps.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = {4096: "tab:gray", 8192: "tab:blue", 16384: "tab:green", 28672: "tab:red"}
    for b in budgets:
        ps = [p for p in pages if f"p{p}|b{b}" in grid and grid[f"p{p}|b{b}"].get("accuracy_mean") is not None]
        x = [pages.index(p) for p in ps]
        axes[0].plot(x, [100 * grid[f"p{p}|b{b}"]["accuracy_mean"] for p in ps], "o-", color=colors.get(b), label=f"budget {b // 1024}K")
        axes[1].plot(x, [100 * grid[f"p{p}|b{b}"]["recall_mean"] for p in ps], "o-", color=colors.get(b))
        axes[2].plot(x, [100 * grid[f"p{p}|b{b}"]["unit_precision_mean"] for p in ps], "o-", color=colors.get(b))
    for ax, t in zip(axes, ("accuracy (%)", "evidence recall (%)", "kept units holding evidence (%)")):
        ax.set_xticks(range(len(pages)))
        ax.set_xticklabels(pages, fontsize=8)
        ax.set_xlabel("page size (tokens)", fontsize=8)
        ax.set_title(t, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("E4 screen: mean over S2 multi-key, BABILong qa2 and qa3 at 128K and 256K (N = 30 per cell)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "e4_page_trend.png", dpi=120)
    plt.close(fig)
    figs.append("figures/e4_page_trend.png")

    if e5["h2"]:
        vs = list(e5["h2"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(vs) + 1.6), sharey=True)
        for ax, L in zip(axes, h2cfg["lengths"]):
            for k, (part, color, off) in enumerate((("multi_fact", "tab:red", -0.15), ("single_fact", "tab:blue", 0.15))):
                ds = [e5["h2"][v]["lengths"][L][part] for v in vs]
                y = np.arange(len(vs)) + off
                ax.errorbar([100 * d["diff"] for d in ds], y, xerr=[[100 * (d["diff"] - d["lo"]) for d in ds], [100 * (d["hi"] - d["diff"]) for d in ds]],
                            fmt="o", color=color, capsize=2, markersize=4, label="multi-fact (qa2+qa3)" if part == "multi_fact" else "single-fact (pooled)")
            ax.axvline(0, color="black", linewidth=0.6)
            ax.axvline(h2cfg["gain_points"], color="tab:red", linewidth=0.6, linestyle=":")
            ax.axvspan(-h2cfg["single_band"], h2cfg["single_band"], color="tab:blue", alpha=0.08)
            ax.set_title(f"{L}: paired difference vs no graph (points, 95% CI)", fontsize=9)
            ax.set_yticks(range(len(vs)))
            ax.set_yticklabels(vs, fontsize=8)
            ax.invert_yaxis()
        axes[0].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / "e5_h2.png", dpi=120)
        plt.close(fig)
        figs.append("figures/e5_h2.png")

    summary = {"commits": commits, "status": status, "prompt_tokens_max": prompt_max, "screen_rows": screen_rows, "screen": grid,
               "page_pairs": page_pairs, "best_settings": best, "e5_setting": e5_setting, "final": final, "final_pairs": final_pairs,
               "a4": a4_fams, "e5": e5, "g1": g1, "ingest": ingest, "figures": figs}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "summary.json", summary)

    # ---------------------------------------------------------------- tables
    out = ["# A5 tables (generated by scripts/a5_summary.py from results/E4_pages and results/E5_graph)", ""]
    out += ["## Rows", "", "| experiment | rows by commit | status | max prompt tokens |", "|---|---|---|---|"]
    for e in exps:
        out.append(f"| {e} | {commits[e]} | {status[e]} | {prompt_max[e]} |")
    out += ["", "## E4 screen: accuracy (%) by page size (rows) and budget (columns), mean of S2 multi-key, qa2, qa3 at 128K and 256K, N = 30 per cell", ""]

    def grid_table(fn, fmt):
        lines = ["| page \\ budget | " + " | ".join(f"{b}" for b in budgets) + " |", "|---" * (len(budgets) + 1) + "|"]
        for p in pages:
            cells_ = []
            for b in budgets:
                g = grid.get(f"p{p}|b{b}")
                v = fn(g) if g else None
                cells_.append(fmt(v) if v is not None else "")
            lines.append(f"| {p} | " + " | ".join(cells_) + " |")
        return lines

    out += grid_table(lambda g: g.get("accuracy_mean"), lambda v: f"{100 * v:.1f}")
    for t in sc["tasks"]:
        out += ["", f"Accuracy (%), {t}, mean of 128K and 256K:", ""]
        out += grid_table(lambda g, t=t: g.get(f"accuracy_{t}"), lambda v: f"{100 * v:.1f}")
    out += ["", "## E4 screen: evidence recall (share of evidence items inside the kept pages), same cells", ""]
    out += grid_table(lambda g: g.get("recall_mean"), lambda v: f"{v:.3f}")
    for t in sc["tasks"]:
        out += ["", f"Evidence recall, {t}, mean of 128K and 256K:", ""]
        out += grid_table(lambda g, t=t: g.get(f"recall_{t}"), lambda v: f"{v:.3f}")
    out += ["", "S3 chain (h3), evidence recall only (no reader), mean of 128K and 256K:", ""]
    out += grid_table(lambda g: g.get("s3_recall_mean"), lambda v: f"{v:.3f}")
    out += ["", "Kept units that hold evidence (unit precision), screen accuracy cells:", ""]
    out += grid_table(lambda g: g.get("unit_precision_mean"), lambda v: f"{v:.3f}")
    out += ["", "Gaps (non-adjacent page boundaries in the kept text), screen accuracy cells:", ""]
    out += grid_table(lambda g: g.get("gaps_mean"), lambda v: f"{v:.1f}")
    out += ["", "Kept tokens, screen accuracy cells:", ""]
    out += grid_table(lambda g: g.get("kept_tokens_mean"), lambda v: f"{v:.0f}")
    out += ["", "## E4 screen: each page size against page 512 at the same budget (paired, pooled over the 6 accuracy cells, 180 pairs)", "",
            "| pair | budget | difference (points) [95% CI] |", "|---|---|---|"]
    for k, d in page_pairs.items():
        pair, b = k.split("|")
        out.append(f"| {pair} | {b[1:]} | {dpct(d)} |")
    if best:
        out += ["", f"## Best settings (rule: {best['rule']})", "", "| rank | page | budget | accuracy | evidence recall |", "|---|---|---|---|---|"]
        for i, r in enumerate(best["ranking"][:8]):
            out.append(f"| {i + 1} | {r['page']} | {r['budget']} | {100 * r['accuracy']:.1f} | {r['evidence_recall']:.3f} |")
    if e5_setting:
        out += ["", f"## E5 setting (rule: {e5_setting['rule']})", "", "| page | budget | score | 256K S1 / S2 / BABILong | 128K S1 / S2 / BABILong |", "|---|---|---|---|---|"]
        for r in e5_setting["ranking"]:
            f256, f128 = r["families"]["256k"], r["families"]["128k"]
            out.append(f"| {r['page']} | {r['budget']} | {100 * r['score']:.1f} | {100 * f256['S1']:.1f} / {100 * f256['S2']:.1f} / {100 * f256['BABILong']:.1f} | "
                       f"{100 * f128['S1']:.1f} / {100 * f128['S2']:.1f} / {100 * f128['BABILong']:.1f} |")
    if final:
        out += ["", "## E4 final: family accuracy (%) [95% CI], N = 100 per task; A4 rows for comparison (M2 hybrid: units of 128 tokens, budget 28,672)", "",
                "| method | " + " | ".join(f"{f} {L}" for f in fams for L in flens) + " |", "|---" * (1 + len(fams) * len(flens)) + "|"]
        for key, rec in final.items():
            out.append(f"| PCM {key} | " + " | ".join(pct(rec["families"].get(f"{f}|{L}")) for f in fams for L in flens) + " |")
        names = {"m2_hybrid": "A4 M2 hybrid (28,672)", "m3_oracle": "A4 M3 oracle", "m3_tight": "A4 M3 tight oracle"}
        for key, nm in names.items():
            out.append(f"| {nm} | " + " | ".join(pct(a4_fams.get(f"{key}|{f}|{L}")) for f in fams for L in flens) + " |")
        out += ["", "Family evidence recall:", "", "| method | " + " | ".join(f"{f} {L}" for f in fams for L in flens) + " |", "|---" * (1 + len(fams) * len(flens)) + "|"]
        for key, rec in final.items():
            out.append(f"| PCM {key} | " + " | ".join(f2((rec["families"].get(f"{f}|{L}") or {}).get("evidence_recall"), 3) for f in fams for L in flens) + " |")
        for key, nm in names.items():
            out.append(f"| {nm} | " + " | ".join(f2((a4_fams.get(f"{key}|{f}|{L}") or {}).get("evidence_recall"), 3) for f in fams for L in flens) + " |")
        out += ["", "Paired differences (points) [95% CI], same examples:", "", "| pair | family | length | difference |", "|---|---|---|---|"]
        for k, d in final_pairs.items():
            pair, f, L = k.split("|")
            out.append(f"| {pair} | {f} | {L} | {dpct(d)} |")
        out += ["", "Per task, accuracy (%) [95% CI] / evidence recall:", "", "| task | length | " + " | ".join(final) + " |", "|---|---|" + "---|" * len(final)]
        keys = sorted({k for rec in final.values() for k in rec["tasks"]}, key=lambda k: (k.split("|")[1], k))
        for k in keys:
            t, L = k.split("|")
            out.append(f"| {t} | {L} | " + " | ".join(f"{pct(rec['tasks'].get(k, {}).get('accuracy'))} / {f2(rec['tasks'].get(k, {}).get('evidence_recall'), 3)}" for rec in final.values()) + " |")
        out += ["", "Cost per example (S1, S2, BABILong, BABILong-ext rows; answers reused from earlier rows count with their own short time):", "",
                "| setting | length | rows | seconds per example | prompt tokens | peak VRAM GiB |", "|---|---|---|---|---|---|"]
        for key, rec in final.items():
            for L, c in rec["cost"].items():
                out.append(f"| {key} | {L} | {c['rows']} | {c['seconds_per_example']} | {c['prompt_tokens']} | {c['peak_vram_gib']} |")
        out += ["", "InfiniteBench, A4 for comparison:", "", "| task | M2 hybrid | M3 oracle | M3 tight |", "|---|---|---|---|"]
        for t in cfg["task_sets"]["infbench"]:
            out.append(f"| {t} | " + " | ".join(f"{pct(a4_fams.get(f'{k}|{t}|full'))} / {f2((a4_fams.get(f'{k}|{t}|full') or {}).get('evidence_recall'), 3)}" for k in names) + " |")
    if e5["variants"]:
        base_name = setting_config(cfg, e5_setting["page"], e5_setting["budget"])["name"]
        for L in ("256k", "128k", "full"):
            vnames = [v for v in e5["variants"] if not v.endswith("-sel") and any(k.endswith(f"|{L}") for k in e5["variants"][v])]
            out += ["", f"## E5 at {L}: accuracy (%) [95% CI]; below it the paired difference against no graph (points) [95% CI]; recall", ""]
            out += ["| task | " + " | ".join(v.replace(base_name, "none" if v == base_name else "").lstrip("-") for v in vnames) + " |", "|---" * (1 + len(vnames)) + "|"]
            tasks = sorted({k.split("|")[0] for v in vnames for k in e5["variants"][v] if k.endswith(f"|{L}")})
            for t in tasks:
                row1, row2 = [], []
                for v in vnames:
                    d = e5["variants"][v].get(f"{t}|{L}")
                    if not d:
                        row1.append("")
                        row2.append("")
                        continue
                    row1.append(f"{pct(d.get('accuracy'))} (n {d['n']})")
                    row2.append(f"{dpct(d.get('diff_vs_none'))}; R {f2(d['evidence_recall'], 3)}" if v != base_name else f"R {f2(d['evidence_recall'], 3)}")
                out.append(f"| {t} | " + " | ".join(row1) + " |")
                out.append("| | " + " | ".join(row2) + " |")
        sel_names = [v for v in e5["variants"] if v.endswith("-sel")]
        if sel_names:
            out += ["", "## E5, S3 chain: evidence recall only (no reader); paired recall difference against no graph", "",
                    "| task | length | " + " | ".join(sel_names) + " |", "|---|---|" + "---|" * len(sel_names)]
            keys = sorted({k for v in sel_names for k in e5["variants"][v]})
            for k in keys:
                t, L = k.split("|")
                cs = []
                for v in sel_names:
                    d = e5["variants"][v].get(k)
                    cs.append("" if not d else f"{f2(d['evidence_recall'], 3)}" + (f" ({dpct(d['recall_diff_vs_none'], 1)})" if d.get("recall_diff_vs_none") else ""))
                out.append(f"| {t} | {L} | " + " | ".join(cs) + " |")
    if e5["h2"]:
        out += ["", f"## H2 (D-045): paired difference against no graph, points [95% CI]; multi-fact = qa2 + qa3 (needs >= +{h2cfg['gain_points']} with CI above 0); "
                f"single-fact pooled (needs |estimate| <= {h2cfg['single_band']})", "",
                "| variant | extractor | length | multi-fact | pass | single-fact (tasks) | pass | secondary multi-fact | recall diff multi / single | holds |", "|---|---|---|---|---|---|---|---|---|---|"]
        for v, r in e5["h2"].items():
            for L in h2cfg["lengths"]:
                x = r["lengths"][L]
                out.append(f"| {v} | {r['extractor']} | {L} | {dpct(x['multi_fact'])} (n {x['multi_fact']['n']}) | {'yes' if x['multi_pass'] else 'no'} | "
                           f"{dpct(x['single_fact'])} (n {x['single_fact']['n']}, {len(x['single_fact']['tasks'])} tasks) | {'yes' if x['single_pass'] else 'no'} | "
                           f"{dpct(x['secondary_multi_fact'])} (n {x['secondary_multi_fact']['n']}) | {f2(x['multi_fact']['recall_diff'], 3)} / {f2(x['single_fact']['recall_diff'], 3)} | "
                           f"{'**yes**' if r['holds'] else 'no'} |")
        out += ["", f"H2 supported (a default-extractor variant holds at both lengths): {'yes' if e5['h2_supported'] else 'no'}; holding variants: {e5['h2_holding_variants'] or 'none'}"]
    if e5["variants"]:
        out += ["", "## E5: how often the graph changes the kept units (examples whose kept units differ from no graph), and question entities found", "",
                "| variant | 128k changed | 256k changed | top-16 seeds moved (256k mean) | question entities: zero share on qa2 / qa3 at 256k |", "|---|---|---|---|---|"]
        for v, rec in e5["variants"].items():
            if v == base_name or v.endswith("-sel"):
                continue
            cs = []
            for L in ("128k", "256k"):
                n = sum(d["n"] for k, d in rec.items() if k.endswith(f"|{L}"))
                same = sum(d.get("same_kept_units", 0) for k, d in rec.items() if k.endswith(f"|{L}"))
                cs.append(f"{n - same} / {n}")
            moved = mean([d.get("moved_in_top_seeds") for k, d in rec.items() if k.endswith("|256k")])
            qz = " / ".join(f2((rec.get(f"{t}|256k") or {}).get("question_entities_zero_share"), 2) for t in ("babilong:qa2", "babilong:qa3"))
            out.append(f"| {v.replace(base_name + '-', '')} | {cs[0]} | {cs[1]} | {f2(moved, 1)} | {qz} |")
    if e5["lengths"]:
        out += ["", "## E5 across lengths: family accuracy (%) [95% CI]; paired difference against no graph", "", "| variant | family | 32k | 64k | 128k | 256k |", "|---|---|---|---|---|---|"]
        for v in ("none", "ppr-all", "hops2"):
            for f in ("S1", "S2", "BABILong", "BABILong-ext"):
                cs = []
                for L in ("32k", "64k", "128k", "256k"):
                    r = e5["lengths"].get(f"{v}|{f}|{L}")
                    cs.append("" if not r else pct(r) + (f"; {dpct(r['diff_vs_none'])}" if "diff_vs_none" in r else ""))
                out.append(f"| {v} | {f} | " + " | ".join(cs) + " |")
    out += ["", f"## Gate G1: evidence recall at {g1c['length']} (threshold {g1c['threshold']} on {', '.join(g1c['tasks'])}; InfiniteBench beside, full length)", "",
            "| setting | " + " | ".join(g1c["tasks"] + g1c["beside"]) + " | pass |", "|---" * (2 + len(g1c["tasks"]) + len(g1c["beside"])) + "|"]
    for name, rec in g1["settings"].items():
        out.append(f"| {name} | " + " | ".join(f"{f2(rec[t]['evidence_recall'], 3)} (all {f2(rec[t]['share_full'], 2)}, n {rec[t]['n']})" for t in g1c["tasks"] + g1c["beside"]) + f" | {'yes' if rec['pass'] else 'no'} |")
    out += ["", "## Ingest and index at 256K (per document)", ""]
    for g in ("synthetic", "babilong"):
        r = ingest.get(g)
        if not r:
            continue
        out.append(f"- {g}: {r.get('docs')} documents, {r.get('tokens')} tokens, {r.get('units')} units; embedding {r.get('embed_s')} s ({r.get('tokens_per_s')} tokens/s); "
                   f"SEM edges {r.get('sem_build_s')} s; default entities + ENT edges {r.get('ent_build_s')} s; spaCy entities {r.get('spacy_entities_s')} s")
        for k, v in (r.get("bytes") or {}).items():
            out.append(f"  - {k}: {v['mean_mib']} MiB ({v['bytes_per_token']} bytes per token, n {v['n']})")
    out += ["", "Query-time selection at 256K (s per example):", "", "| variant | n | select_s | graph_s | prepare_s |", "|---|---|---|---|---|"]
    for v, r in qcost.items():
        out.append(f"| {v} | {r['n']} | {r['select_s']} | {r['graph_s']} | {r['prepare_s']} |")
    atomic_write_text(OUT / "tables.md", "\n".join(out) + "\n")
    print(f"wrote {OUT / 'summary.json'} and tables.md; figures: {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
