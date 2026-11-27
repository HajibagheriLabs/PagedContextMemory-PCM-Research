"""Numbers, tables and figures for reports/A6_kv.md.

    python scripts/a6_summary.py

Reads results/E6_kv (runs.jsonl, ape_choice.json, deviation.jsonl, store_bench.jsonl) and results/E6_kv_tune. Text-mode
rows of E6 reuse A5's answers to identical prompts; their timings are read from the A5 rows that produced the answer
(results/E4_pages, results/E5_graph). How G2 and H3 are read is D-051. Writes results/A6_kv/summary.json, tables.md
and figures/*.png. Every table gives n.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcm.eval import metrics as M  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json, atomic_write_text  # noqa: E402
from pcm.tasks.common import config_hash, load_task_config  # noqa: E402
from run_a5 import expand_tasks  # noqa: E402
from run_a6 import load_config, mode_config  # noqa: E402

OUT = REPO_ROOT / "results" / "A6_kv"
FIG = OUT / "figures"
LENGTHS = ["32k", "64k", "128k", "256k"]


def rows_of(exp: str, name: str = "runs.jsonl") -> list[dict]:
    p = REPO_ROOT / "results" / exp / name
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


def main() -> int:
    tcfg = load_task_config()
    cfg = load_config()
    mc = tcfg["metrics"]
    B, SEED, CI = mc["bootstrap_resamples"], mc["bootstrap_seed"], mc["ci"]

    def ci(xs):
        return M.bootstrap_ci(list(xs), B, SEED, CI)

    def paired(a, b):
        return M.paired_bootstrap(a, b, B, SEED, CI)

    ape_path = REPO_ROOT / "results" / "E6_kv" / "ape_choice.json"
    ape = json.loads(ape_path.read_text(encoding="utf-8")) if ape_path.is_file() else None
    rows = rows_of("E6_kv")
    commits = dict(Counter(f"{(r.get('git_commit') or '')[:10]}{'+dirty' if r.get('git_dirty') else ''}" for r in rows))
    status = dict(Counter(r["status"] for r in rows))
    source = {}
    for exp in ("E4_pages", "E5_graph"):
        for r in rows_of(exp):
            # E5's no-graph rows share ids with E4's final rows (same method hash) and reuse their answers: keep the
            # row that produced the answer.
            if r["id"] in source and not (source[r["id"]].get("timings") or {}).get("reused"):
                continue
            source[r["id"]] = r

    cells = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            cells[(r["method_hash"], r["task"], r["length"])].append(r)

    modes = {"text": mode_config(cfg, "text"), "kv": mode_config(cfg, "kv")}
    if ape:
        modes["kv-ape"] = mode_config(cfg, "kv-ape", ape=ape)

    def cell(m, task, L):
        return cells.get((config_hash(m), task, L), [])

    def acc(rs):
        return [r["score"] for r in rs if r.get("score") is not None]

    def pair(ra, rb):
        return M.paired_by_id([r for r in ra if r.get("score") is not None], [r for r in rb if r.get("score") is not None])

    def origin(r):
        """The row that produced a reused answer, following reuse across experiments (E6 -> E5 -> E4)."""
        seen = 0
        while (r.get("timings") or {}).get("reused") and r.get("reused_from") in source and seen < 5:
            r = source[r["reused_from"]]
            seen += 1
        return r

    def text_timing(r, key):
        """A text-mode timing; reused answers take it from the A5 row that produced the answer."""
        return (origin(r).get("timings") or {}).get(key)

    retained = expand_tasks(cfg, ["h1", "ext"])
    infb = cfg["task_sets"]["infbench"]
    fams = cfg["families"]

    # ---- accuracy per mode, task, length
    table = {}
    for mname, m in modes.items():
        for t in retained:
            for L in LENGTHS:
                rs = cell(m, t, L)
                if rs:
                    table[f"{mname}|{t}|{L}"] = {"n": len(rs), "accuracy": ci(acc(rs))}
        for t in infb:
            rs = cell(m, t, "full")
            if rs:
                table[f"{mname}|{t}|full"] = {"n": len(rs), "accuracy": ci(acc(rs))}

    # ---- gaps (D-051): KV - text, pooled over the retained tasks, per length; per family; InfiniteBench
    gaps, fam_gaps = {}, {}
    for kvm in [k for k in ("kv", "kv-ape") if k in modes]:
        for L in LENGTHS:
            a, b = [], []
            for t in retained:
                x, y, _ = pair(cell(modes[kvm], t, L), cell(modes["text"], t, L))
                a += x
                b += y
            if a:
                g = paired(a, b)
                g.update({"kv_mean": mean(a), "text_mean": mean(b)})
                gaps[f"{kvm}|{L}"] = g
            for f, tasks in fams.items():
                a, b = [], []
                for t in tasks:
                    x, y, _ = pair(cell(modes[kvm], t, L), cell(modes["text"], t, L))
                    a += x
                    b += y
                if a:
                    g = paired(a, b)
                    g.update({"kv_mean": mean(a), "text_mean": mean(b)})
                    fam_gaps[f"{kvm}|{f}|{L}"] = g
        for t in infb:
            x, y, _ = pair(cell(modes[kvm], t, "full"), cell(modes["text"], t, "full"))
            if x:
                g = paired(x, y)
                g.update({"kv_mean": mean(x), "text_mean": mean(y)})
                fam_gaps[f"{kvm}|{t}|full"] = g

    # ---- tokens prefilled at query time, time to first token, expansion, peak VRAM
    speed = {}
    for mname, m in modes.items():
        for L in LENGTHS + ["full"]:
            tasks = infb if L == "full" else retained
            rs = [r for t in tasks for r in cell(m, t, L)]
            if not rs:
                continue
            if mname == "text":
                rec = {"rows": len(rs), "query_prefill_tokens": mean([r.get("prompt_tokens") for r in rs]),
                       "ttft_s": mean([text_timing(r, "prefill_s") for r in rs]), "expand_ms": None, "encode_s": None,
                       "timed_rows": sum(1 for r in rs if text_timing(r, "prefill_s") is not None),
                       "peak_vram_gib": max([origin(r).get("peak_vram_gib") or 0 for r in rs])}
            else:
                fresh = [r for r in rs if not (r.get("timings") or {}).get("reused")]
                rec = {"rows": len(rs), "timed_rows": len(fresh), "query_prefill_tokens": mean([(r.get("timings") or {}).get("query_prefill_tokens") for r in rs]),
                       "ttft_s": mean([(r["timings"].get("expand_s") or 0) + (r["timings"].get("prefill_s") or 0) for r in fresh]),
                       "expand_ms": rnd(1000 * mean([r["timings"].get("expand_s") for r in fresh]), 1) if fresh else None,
                       "encode_s": mean([r["timings"].get("kv_encode_s") for r in fresh]),
                       "stitched_boundaries": mean([(r.get("selection") or {}).get("stitched_boundaries") for r in rs]),
                       "peak_vram_gib": max([r.get("peak_vram_gib") or 0 for r in fresh] or [0])}
            speed[f"{mname}|{L}"] = {k: (rnd(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}
    ratios = {}
    for L in LENGTHS:
        t, k = speed.get(f"text|{L}"), speed.get(f"kv|{L}")
        if t and k and k["query_prefill_tokens"]:
            ratios[L] = rnd(t["query_prefill_tokens"] / k["query_prefill_tokens"], 2)

    # ---- G2 and H3 (D-051)
    best = "kv-ape" if "kv-ape" in modes else "kv"
    verdict = {}
    for kvm in [k for k in ("kv", "kv-ape") if k in modes]:
        ok = [L for L in LENGTHS if f"{kvm}|{L}" in gaps]
        g2 = len(ok) == len(LENGTHS) and all(gaps[f"{kvm}|{L}"]["diff"] >= -0.05 - 1e-9 for L in ok)
        h3 = g2 and all(gaps[f"{kvm}|{L}"]["diff"] >= -0.03 - 1e-9 for L in ok) and all((ratios.get(L) or 0) >= cfg["h3_prefill_ratio"] for L in ok)
        verdict[kvm] = {"lengths": ok, "g2_pass": bool(g2), "h3_holds": bool(h3)}
    verdict["best_fix"] = best

    # ---- (d) page sizes: KV (with the fix) - text at each page size, pooled over the page tasks, per length
    pages = {}
    pg = cfg["e6"]["pages"]
    for size in pg["sizes"]:
        tm = mode_config(cfg, "text", page=size)
        km = mode_config(cfg, "kv-ape", page=size, ape=ape) if ape else None
        if km is None:
            continue
        for L in pg["lengths"]:
            a, b, bnd = [], [], []
            for t in expand_tasks(cfg, pg["tasks"]):
                kr, tr = cell(km, t, L), cell(tm, t, L)
                x, y, _ = pair(kr, tr)
                a += x
                b += y
                bnd += [(r.get("selection") or {}).get("stitched_boundaries") for r in kr]
            if a:
                g = paired(a, b)
                g.update({"kv_mean": mean(a), "text_mean": mean(b), "stitched_boundaries": rnd(mean(bnd), 1)})
                pages[f"{size}|{L}"] = g

    # ---- gap by the number of stitched boundaries within the main comparison (observational)
    by_bnd = {}
    if best in modes:
        pairs = defaultdict(lambda: ([], []))
        for L in LENGTHS:
            for t in retained:
                kr = {r["example_id"]: r for r in cell(modes[best], t, L)}
                tr = {r["example_id"]: r for r in cell(modes["text"], t, L)}
                for eid in sorted(set(kr) & set(tr)):
                    nb = (kr[eid].get("selection") or {}).get("stitched_boundaries")
                    if nb is None:
                        continue
                    key = "<= 20" if nb <= 20 else "21-25" if nb <= 25 else "26-30" if nb <= 30 else "> 30"
                    pairs[key][0].append(kr[eid]["score"])
                    pairs[key][1].append(tr[eid]["score"])
        for k, (a, b) in pairs.items():
            by_bnd[k] = {**paired(a, b), "kv_mean": mean(a), "text_mean": mean(b)}

    # ---- deviation by layer
    dev = rows_of("E6_kv", "deviation.jsonl")
    deviation = {}
    if dev:
        nl = len(dev[0]["kv_deviation"])
        deviation = {"n": len(dev), "tasks": dict(Counter(r["task"] for r in dev)),
                     "k": [rnd(mean([r["kv_deviation"][i]["k"] for r in dev])) for i in range(nl)],
                     "v": [rnd(mean([r["kv_deviation"][i]["v"] for r in dev])) for i in range(nl)],
                     "attention_tv": [rnd(mean([r["attention_deviation"][i] for r in dev])) for i in range(nl)],
                     "kl": rnd(mean([r["next_token"]["kl"] for r in dev]), 5), "top1_same": rnd(mean([1.0 if r["next_token"]["top1_same"] else 0.0 for r in dev]), 3),
                     "pages": rnd(mean([r["pages"] for r in dev]), 1), "segments": rnd(mean([r["segments"] for r in dev]), 1)}
        withfix = [r for r in dev if "ape" in r]
        if withfix:
            deviation["ape"] = {"n": len(withfix), "attention_tv": [rnd(mean([r["ape"]["attention_deviation"][i] for r in withfix])) for i in range(nl)],
                                "kl": rnd(mean([r["ape"]["next_token"]["kl"] for r in withfix]), 5),
                                "top1_same": rnd(mean([1.0 if r["ape"]["next_token"]["top1_same"] else 0.0 for r in withfix]), 3)}
        deviation["commits"] = dict(Counter((r.get("git_commit") or "")[:10] for r in dev))

    store = rows_of("E6_kv", "store_bench.jsonl")

    # ---- tuning grid
    tune = rows_of("E6_kv_tune")
    grid = {}
    for t in cfg["ape"]["temperatures"]:
        for s in cfg["ape"]["scales"]:
            h = config_hash(mode_config(cfg, "kv-ape", ape={"temperature": t, "scale": s}, suffix=f"-T{t:g}-S{s:g}-train"))
            per = {}
            for task in cfg["ape"]["train"]["tasks"]:
                sc = [r["score"] for r in tune if r["method_hash"] == h and r["task"] == task and r["status"] == "ok"]
                per[task] = {"n": len(sc), "accuracy": rnd(mean(sc))}
            grid[f"T{t:g}|S{s:g}"] = {"temperature": t, "scale": s, "tasks": per, "mean": rnd(mean([v["accuracy"] for v in per.values()]))}
    tune_commits = dict(Counter((r.get("git_commit") or "")[:10] for r in tune))

    # ---- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    figs = []
    if gaps:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
        for kvm, off, color in (("kv", -0.12, "tab:gray"), ("kv-ape", 0.12, "tab:blue")):
            xs = [i for i, L in enumerate(LENGTHS) if f"{kvm}|{L}" in gaps]
            if not xs:
                continue
            g = [gaps[f"{kvm}|{LENGTHS[i]}"] for i in xs]
            axes[0].errorbar([x + off for x in xs], [100 * d["diff"] for d in g], yerr=[[100 * (d["diff"] - d["lo"]) for d in g], [100 * (d["hi"] - d["diff"]) for d in g]],
                             fmt="o", color=color, capsize=3, label="KV plain" if kvm == "kv" else "KV + APE fix")
        axes[0].axhline(0, color="black", linewidth=0.6)
        axes[0].axhline(-5, color="tab:red", linewidth=0.8, linestyle=":")
        axes[0].axhline(-3, color="tab:orange", linewidth=0.8, linestyle=":")
        axes[0].set_xticks(range(len(LENGTHS)))
        axes[0].set_xticklabels(LENGTHS)
        axes[0].set_ylabel("KV - text (points)")
        axes[0].set_title("Gap, pooled over S1, S2, BABILong qa1-qa10 (95% CI). Dotted: -5 (G2), -3 (H3)", fontsize=8)
        axes[0].legend(fontsize=7)
        for L, color in (("128k", "tab:blue"), ("256k", "tab:green")):
            ks = [s for s in pg["sizes"] if f"{s}|{L}" in pages]
            if ks:
                d = [pages[f"{s}|{L}"] for s in ks]
                axes[1].errorbar([x["stitched_boundaries"] for x in d], [100 * x["diff"] for x in d], yerr=[[100 * (x["diff"] - x["lo"]) for x in d], [100 * (x["hi"] - x["diff"]) for x in d]],
                                 fmt="o-", color=color, capsize=3, label=L)
                for s, x in zip(ks, d):
                    axes[1].annotate(f"page {s}", (x["stitched_boundaries"], 100 * x["diff"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        axes[1].set_xscale("log")
        axes[1].axhline(0, color="black", linewidth=0.6)
        axes[1].set_xlabel("stitched boundaries per prompt (log)")
        axes[1].set_title("KV + fix - text by page size (same budget), S2 mk/mv, qa1, qa2", fontsize=8)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "gaps.png", dpi=120)
        plt.close(fig)
        figs.append("figures/gaps.png")
    if deviation:
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
        x = np.arange(len(deviation["k"]))
        axes[0].plot(x, deviation["k"], "o-", label="keys")
        axes[0].plot(x, deviation["v"], "s-", label="values")
        axes[0].set_xlabel("layer")
        axes[0].set_ylabel("relative deviation")
        axes[0].set_title(f"KV deviation, stitched vs sequential (n = {deviation['n']})", fontsize=9)
        axes[0].legend(fontsize=7)
        axes[1].plot(x, deviation["attention_tv"], "o-", label="plain")
        if "ape" in deviation:
            axes[1].plot(x, deviation["ape"]["attention_tv"], "s-", label="with APE fix")
        axes[1].set_xlabel("layer")
        axes[1].set_ylabel("total variation")
        axes[1].set_title("Attention deviation of the question tokens", fontsize=9)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "deviation.png", dpi=120)
        plt.close(fig)
        figs.append("figures/deviation.png")

    summary = {"commits": commits, "status": status, "ape": ape, "tune": grid, "tune_commits": tune_commits, "table": table, "gaps": gaps,
               "family_gaps": fam_gaps, "speed": speed, "prefill_ratio": ratios, "verdict": verdict, "pages": pages, "by_boundaries": by_bnd,
               "deviation": deviation, "store": store, "figures": figs}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "summary.json", summary)

    # ---- tables
    out = ["# A6 tables (generated by scripts/a6_summary.py)", "", f"Rows: {status}; commits {commits}", ""]
    out += ["## APE tuning on the synthetic train split (128K, 30 per task): mean accuracy", "", "| T \\ S | " + " | ".join(f"{s:g}" for s in cfg["ape"]["scales"]) + " |",
            "|---" * (1 + len(cfg["ape"]["scales"])) + "|"]
    for t in cfg["ape"]["temperatures"]:
        out.append(f"| {t:g} | " + " | ".join(f2(100 * grid[f'T{t:g}|S{s:g}']['mean'], 1) if grid[f'T{t:g}|S{s:g}']['mean'] is not None else "–" for s in cfg["ape"]["scales"]) + " |")
    if ape:
        out += ["", f"Choice: T = {ape['temperature']}, S = {ape['scale']} ({ape['rule']})"]
    out += ["", "## Gap KV - text, pooled over the 13 retained tasks (D-051), points [95% CI]", "", "| mode | " + " | ".join(LENGTHS) + " |", "|---" * 5 + "|"]
    for kvm in ("kv", "kv-ape"):
        if any(f"{kvm}|{L}" in gaps for L in LENGTHS):
            out.append(f"| {kvm} | " + " | ".join(f"{dpct(gaps.get(f'{kvm}|{L}'))} (n {gaps.get(f'{kvm}|{L}', {}).get('n', 0)}; text {f2(100 * (gaps.get(f'{kvm}|{L}', {}).get('text_mean') or 0), 1)}, KV {f2(100 * (gaps.get(f'{kvm}|{L}', {}).get('kv_mean') or 0), 1)})" for L in LENGTHS) + " |")
    out += ["", f"Verdict: {verdict}; query prefill ratio text/KV: {ratios}"]
    out += ["", "## Gap by family, points [95% CI]", "", "| mode | family | " + " | ".join(LENGTHS) + " |", "|---" * 6 + "|"]
    for kvm in ("kv", "kv-ape"):
        for f in fams:
            if any(f"{kvm}|{f}|{L}" in fam_gaps for L in LENGTHS):
                out.append(f"| {kvm} | {f} | " + " | ".join(f"{dpct(fam_gaps.get(f'{kvm}|{f}|{L}'))}; text {f2(100 * (fam_gaps.get(f'{kvm}|{f}|{L}', {}).get('text_mean') or 0), 1)}" for L in LENGTHS) + " |")
    out += ["", "InfiniteBench (own length):", "", "| mode | task | text | KV | gap |", "|---|---|---|---|---|"]
    for kvm in ("kv", "kv-ape"):
        for t in infb:
            g = fam_gaps.get(f"{kvm}|{t}|full")
            if g:
                out.append(f"| {kvm} | {t} | {100 * g['text_mean']:.1f} | {100 * g['kv_mean']:.1f} | {dpct(g)} (n {g['n']}) |")
    out += ["", "## Accuracy per task (%), text / KV / KV + fix", "", "| task | " + " | ".join(LENGTHS) + " |", "|---" * 5 + "|"]
    for t in retained:
        cs = []
        for L in LENGTHS:
            vals = [table.get(f"{m}|{t}|{L}") for m in ("text", "kv", "kv-ape")]
            cs.append(" / ".join(f2(100 * v["accuracy"]["mean"], 0) if v else "–" for v in vals))
        out.append(f"| {t} | " + " | ".join(cs) + " |")
    out += ["", "## Speed per example (mean over the retained tasks; InfiniteBench at its own length)", "",
            "| mode | length | rows | tokens prefilled at query time | time to first token s | expansion ms | page encoding s (ingest) | stitched boundaries | peak VRAM GiB |", "|---" * 9 + "|"]
    for k, v in speed.items():
        mname, L = k.split("|")
        out.append(f"| {mname} | {L} | {v['rows']} | {f2(v['query_prefill_tokens'], 0)} | {f2(v['ttft_s'], 3)} | {f2(v.get('expand_ms'), 1)} | {f2(v.get('encode_s'), 2)} | "
                   f"{f2(v.get('stitched_boundaries'), 1)} | {f2(v['peak_vram_gib'], 2)} |")
    out += ["", "## (d) Page size: KV + fix - text on the same pages, S2 multi-key, S2 multi-value, qa1, qa2, points [95% CI]", "",
            "| page | length | stitched boundaries | text | KV + fix | gap |", "|---|---|---|---|---|---|"]
    for k, g in pages.items():
        s, L = k.split("|")
        out.append(f"| {s} | {L} | {g['stitched_boundaries']} | {100 * g['text_mean']:.1f} | {100 * g['kv_mean']:.1f} | {dpct(g)} (n {g['n']}) |")
    out += ["", "Gap by stitched boundaries within the main comparison (all lengths, KV + fix):", "", "| boundaries | n | text | KV | gap |", "|---|---|---|---|---|"]
    for k in ("<= 20", "21-25", "26-30", "> 30"):
        if k in by_bnd:
            g = by_bnd[k]
            out.append(f"| {k} | {g['n']} | {100 * g['text_mean']:.1f} | {100 * g['kv_mean']:.1f} | {dpct(g)} |")
    if deviation:
        out += ["", f"## Deviation of stitched against sequential caches (n = {deviation['n']}, {deviation['tasks']}, 128K; {deviation['pages']} pages, {deviation['segments']} segments)", "",
                "| layer | K | V | attention TV | attention TV with fix |", "|---|---|---|---|---|"]
        for i in range(len(deviation["k"])):
            fix = deviation.get("ape", {}).get("attention_tv", [None] * len(deviation["k"]))[i]
            out.append(f"| {i} | {deviation['k'][i]:.4f} | {deviation['v'][i]:.4f} | {deviation['attention_tv'][i]:.4f} | {f2(fix, 4)} |")
        out += ["", f"Next token at the first answer position: KL {deviation['kl']}, top-1 same {deviation['top1_same']}" +
                (f"; with fix KL {deviation['ape']['kl']}, top-1 same {deviation['ape']['top1_same']}" if "ape" in deviation else "")]
    if store:
        out += ["", "## Whole-document page store at 256K", "", "| example | pages | ingest s | tokens/s | GPU GiB | disk GiB | disk write s | selected on disk | expand ms | recompute ms | peak GiB |",
                "|---" * 11 + "|"]
        for r in store:
            out.append(f"| {r['id']} | {r['pages']} | {r['ingest_s']:.1f} | {r['ingest_tokens_per_s']:.0f} | {r['store_gpu_bytes'] / 1024**3:.2f} | {r['store_disk_bytes'] / 1024**3:.2f} | "
                       f"{r['disk_write_s']:.1f} | {r['selected_on_disk']} of {r['selected_pages']} | {1000 * r['expand_s']:.0f} | {1000 * r['recompute_s']:.0f} | {r['peak_vram_gib']:.1f} |")
    atomic_write_text(OUT / "tables.md", "\n".join(out) + "\n")
    print(f"wrote {OUT / 'summary.json'}, tables.md; figures {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
