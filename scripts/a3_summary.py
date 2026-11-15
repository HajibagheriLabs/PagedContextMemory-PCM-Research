"""Numbers and figures for reports/A3_floor.md, from results/E0_floor and results/E1_truncation.

    python scripts/a3_summary.py

Writes results/A3_floor/summary.json and results/A3_floor/figures/*.png and prints the
tables. Reads the last row per id of each runs.jsonl; only rows with status "ok" score.
Gate G0 follows configs/a3_floor.yaml (gate) as interpreted in D-024.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.eval import metrics as M  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.common import load_task_config  # noqa: E402

OUT = REPO_ROOT / "results" / "A3_floor"
FIG = OUT / "figures"
ORDER = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "full"]


def rows_of(exp: str) -> list[dict]:
    last: dict = {}
    for r in JsonlLog(REPO_ROOT / "results" / exp / "runs.jsonl").rows():
        last[r["id"]] = r
    return list(last.values())


def chance(task: str, tcfg: dict) -> float:
    bench, name = task.split(":")
    return float(tcfg["synth"]["tasks"][name]["chance"]) if bench == "synth" else float(tcfg[bench]["chance"][name])


def ci(scores: list[float], m: dict) -> dict:
    return M.bootstrap_ci(scores, m["bootstrap_resamples"], m["bootstrap_seed"], m["ci"])


def pct(c: dict | None) -> str:
    if not c or c.get("mean") is None:
        return "–"
    return f"{100 * c['mean']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def main() -> int:
    tcfg = load_task_config()
    acfg = yaml.safe_load((REPO_ROOT / "configs" / "a3_floor.yaml").read_text(encoding="utf-8"))
    mcfg = tcfg["metrics"]
    rows = rows_of("E0_floor") + rows_of("E1_truncation")
    commits = Counter(((r.get("git_commit") or "")[:10], bool(r.get("git_dirty"))) for r in rows)

    cells: dict = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["task"], r["length"])].append(r)
    table = {}
    for (m, t, L), rs in cells.items():
        ok = [r for r in rs if r["status"] == "ok"]
        kinds = Counter(M.error_kind(r["prediction"], r) for r in ok)
        rec = [r["evidence_recall"] for r in ok if r.get("evidence_recall") is not None]
        table[f"{m}|{t}|{L}"] = {
            "method": m, "task": t, "length": L, "accuracy": ci([r["score"] for r in ok], mcfg),
            "status": dict(Counter(r["status"] for r in rs)), "error_kinds": dict(kinds),
            "tokens_mean": round(sum(r["length_tokens"] for r in ok) / len(ok), 1) if ok else None,
            "prompt_tokens_mean": round(sum(r["prompt_tokens"] for r in ok) / len(ok), 1) if ok else None,
            "evidence_recall": round(sum(rec) / len(rec), 4) if rec else None,
            "seconds_per_example": round(sum(r["timings"]["total_s"] for r in ok) / len(ok), 3) if ok else None,
            "peak_vram_gib": round(max(r.get("peak_vram_gib") or 0 for r in rs), 3),
        }
        if t.startswith("babilong:"):  # supplementary, beside the official score (D-028)
            table[f"{m}|{t}|{L}"]["accuracy_whole_word"] = ci([M.babilong_whole_word_score(r["prediction"], r) for r in ok], mcfg)

    def acc(m, t, L):
        c = table.get(f"{m}|{t}|{L}")
        return c["accuracy"]["mean"] if c else None

    # ---- Gate G0
    g = acfg["gate"]

    def task_threshold(t: str) -> float:
        """>= 50%, or >= 2 x chance when that is lower; a task with chance 0 needs 50% (D-027)."""
        c = chance(t, tcfg)
        return min(g["pass_accuracy"], g["pass_chance_multiple"] * c) if c > 0 else g["pass_accuracy"]

    gate = {}
    for reader in ("native-1.7B", "native-0.6B"):
        fams = {}
        for fam, tasks in g["families"].items():
            accs = [acc(reader, t, g["length"]) for t in tasks]
            if any(a is None for a in accs):
                fams[fam] = {"complete": False}
                continue
            a = sum(accs) / len(accs)
            ch = sum(chance(t, tcfg) for t in tasks) / len(tasks)
            per_task = {t: {"accuracy": acc(reader, t, g["length"]), "chance": chance(t, tcfg), "threshold": task_threshold(t)}
                        for t in tasks}
            for v in per_task.values():
                v["passes"] = v["accuracy"] >= v["threshold"] - M.ECL_EPS
            threshold = sum(v["threshold"] for v in per_task.values()) / len(per_task)  # D-027
            fams[fam] = {"complete": True, "accuracy": a, "chance": ch, "threshold": threshold,
                         "passes": a >= threshold - M.ECL_EPS, "tasks": per_task}
        passing = [f for f, v in fams.items() if v.get("passes")]
        gate[reader] = {"families": fams, "passing": passing, "gate_passes": len(passing) >= g["min_families"]}

    # ---- references for ECL90: native-1.7B at 8K, per task and per family
    reference = {"per_task": {}, "per_family": {}}
    for fam, tasks in g["families"].items():
        for t in tasks:
            reference["per_task"][t] = table.get(f"native-1.7B|{t}|8k", {}).get("accuracy")
        f = gate["native-1.7B"]["families"].get(fam, {})
        reference["per_family"][fam] = f.get("accuracy")

    # ---- paired reader comparison at 8K and below (identical examples)
    def scores_by_id(m, t, L):
        return {r["example_id"]: r["score"] for r in cells.get((m, t, L), []) if r["status"] == "ok"}

    readers = {}
    for fam, tasks in g["families"].items():
        lengths = ["8k"] if tasks[0].startswith("synth") else ["0k", "1k", "2k", "4k", "8k"]
        for L in lengths:
            a_all, b_all = [], []
            for t in tasks:
                a, b = scores_by_id("native-1.7B", t, L), scores_by_id("native-0.6B", t, L)
                ids = sorted(set(a) & set(b))
                a_all += [a[i] for i in ids]
                b_all += [b[i] for i in ids]
                readers[f"{t}|{L}"] = M.paired_bootstrap([a[i] for i in ids], [b[i] for i in ids], mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"])
            readers[f"family:{fam}|{L}"] = M.paired_bootstrap(a_all, b_all, mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"])

    # ---- greedy versus sampling on the 50-example subset
    samp = [r for r in rows if r["method"] == "sampled-1.7B" and r["status"] == "ok"]
    greedy = {(r["task"], r["example_id"]): r["score"] for r in rows if r["method"] == "native-1.7B" and r["status"] == "ok"}
    pairs = [(greedy[(r["task"], r["example_id"])], r["score"]) for r in samp if (r["task"], r["example_id"]) in greedy]
    sampling = {"n": len(pairs), "greedy": ci([p[0] for p in pairs], mcfg), "sampled": ci([p[1] for p in pairs], mcfg),
                "paired_sampled_minus_greedy": M.paired_bootstrap([p[1] for p in pairs], [p[0] for p in pairs], mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"]),
                "changed_answers": sum(1 for r in samp if (r["task"], r["example_id"]) in greedy and r["score"] != greedy[(r["task"], r["example_id"])])}

    # ---- depth deciles at 32K (synthetic tasks with a decile)
    deciles = {}
    for t in acfg["task_sets"]["synth"]:
        ok = [r for r in cells.get(("native-1.7B", t, "32k"), []) if r["status"] == "ok"]
        by = M.accuracy_by_decile(ok, mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"])
        if by:
            deciles[t] = {str(d): v for d, v in by.items()}

    # ---- ECL90 of the native reader + M0 curve (native up to 32K, M0 beyond), per family
    grid = list(tcfg["lengths"])
    ecl = {}
    for m0 in ("m0-tail-1.7B", "m0-headtail-1.7B"):
        for fam, tasks in g["families"].items():
            curve = {}
            for L in grid:
                src = "native-1.7B" if L in ("8k", "16k", "32k") else m0
                accs = [acc(src, t, L) for t in tasks]
                if all(a is not None for a in accs):
                    curve[L] = sum(accs) / len(accs)
            ref = reference["per_family"].get(fam)
            ch = sum(chance(t, tcfg) for t in tasks) / len(tasks)
            if ref is not None:
                ecl[f"{m0}|{fam}"] = {**M.ecl90(curve, ref, ch, grid, mcfg["ecl_ratio"], mcfg["ecl_floor_min"]), "curve": curve}

    # ---- typical errors: up to 3 wrong and 3 format predictions per task (1.7B native, 8K)
    examples = {}
    for t in acfg["task_sets"]["synth"] + acfg["task_sets"]["babilong"]:
        ok = [r for r in cells.get(("native-1.7B", t, "8k"), []) if r["status"] == "ok"]
        ex = {"wrong": [], "format": []}
        for r in ok:
            k = M.error_kind(r["prediction"], r)
            if k in ex and len(ex[k]) < 3:
                ex[k].append({"id": r["example_id"], "prediction": r["prediction"][:200], "answers": r["answers"]})
        examples[t] = ex

    # ---- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    figs = []
    if deciles:
        fig, ax = plt.subplots(figsize=(7, 4))
        for t, by in deciles.items():
            xs = sorted(int(d) for d in by)
            ax.plot(xs, [100 * by[str(d)]["mean"] for d in xs], marker="o", label=t.split(":")[1])
        ax.set_xlabel("evidence depth decile (0 = start of the context)")
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(-2, 102)
        ax.set_xticks(range(10))
        ax.set_title("Qwen3-1.7B native at 32K: accuracy by evidence depth (n = 10 per point)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / "deciles_32k.png", dpi=120)
        plt.close(fig)
        figs.append("figures/deciles_32k.png")
    tasks_all = acfg["task_sets"]["synth"] + acfg["task_sets"]["babilong"]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharey=True)
    for ax, t in zip(axes.flat, tasks_all):
        for m, style in (("native-1.7B", "o-"), ("native-0.6B", "s--"), ("m0-tail-1.7B", "^-"), ("m0-headtail-1.7B", "v-")):
            Ls = [L for L in ORDER if f"{m}|{t}|{L}" in table and table[f"{m}|{t}|{L}"]["accuracy"]["mean"] is not None]
            if not Ls:
                continue
            ys = [100 * table[f"{m}|{t}|{L}"]["accuracy"]["mean"] for L in Ls]
            lo = [100 * (table[f"{m}|{t}|{L}"]["accuracy"]["mean"] - table[f"{m}|{t}|{L}"]["accuracy"]["lo"]) for L in Ls]
            hi = [100 * (table[f"{m}|{t}|{L}"]["accuracy"]["hi"] - table[f"{m}|{t}|{L}"]["accuracy"]["mean"]) for L in Ls]
            ax.errorbar([ORDER.index(L) for L in Ls], ys, yerr=[lo, hi], fmt=style, capsize=2, markersize=3, label=m)
        ax.set_title(t, fontsize=8)
        ax.set_xticks(range(len(ORDER) - 1))
        ax.set_xticklabels(ORDER[:-1], fontsize=6, rotation=45)
        ax.set_ylim(-2, 102)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Accuracy (%) vs length: native readers up to 32K, truncation M0 at 64K-256K (95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "curves.png", dpi=110)
    plt.close(fig)
    figs.append("figures/curves.png")

    summary = {"commits": {f"{c}{'+dirty' if d else ''}": k for (c, d), k in commits.items()}, "cells": table, "gate": gate,
               "reference": reference, "readers": readers, "sampling": sampling, "deciles_32k": deciles, "ecl90": ecl,
               "typical_errors": examples, "figures": figs}
    atomic_write_json(OUT / "summary.json", summary)

    # ---- print
    print("commits:", summary["commits"])
    for reader in gate:
        print(f"\n## Gate G0 ({reader} at 8K): passing families {gate[reader]['passing']} -> gate {'PASSES' if gate[reader]['gate_passes'] else 'FAILS'}")
        for fam, v in gate[reader]["families"].items():
            if v.get("complete"):
                print(f"| {fam} | {100 * v['accuracy']:.1f} | chance {100 * v['chance']:.1f} | threshold {100 * v['threshold']:.1f} | {'pass' if v['passes'] else 'fail'} | "
                      + ", ".join(f"{t.split(':')[1]} {100 * x['accuracy']:.0f}{'' if x['passes'] else '(f)'}" for t, x in v["tasks"].items()) + " |")
    print("\n## Cells")
    for k in sorted(table, key=lambda k: (k.split("|")[0], k.split("|")[1], ORDER.index(k.split("|")[2]))):
        c = table[k]
        print(f"| {c['method']} | {c['task']} | {c['length']} | {c['status']} | {pct(c['accuracy'])} | {c['error_kinds']} | tok {c['tokens_mean']} | rec {c['evidence_recall']} | {c['seconds_per_example']} s | {c['peak_vram_gib']} GiB |")
    print("\n## BABILong, official vs whole-word labels (D-028)")
    for k in sorted(table):
        c = table[k]
        if "accuracy_whole_word" in c and c["accuracy"]["mean"] != c["accuracy_whole_word"]["mean"]:
            print(f"| {c['method']} | {c['task']} | {c['length']} | official {pct(c['accuracy'])} | whole-word {pct(c['accuracy_whole_word'])} |")
    print("\n## Readers (1.7B - 0.6B, paired)")
    for k, v in readers.items():
        if k.startswith("family:") and v["n"]:
            print(f"| {k} | n {v['n']} | {100 * v['diff']:+.1f} [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}] | excludes 0: {v['excludes_0']} |")
    print("\n## Sampling vs greedy:", json.dumps(sampling))
    print("\n## ECL90 (native to 32K, then M0):")
    for k, v in ecl.items():
        print(f"| {k} | ref {100 * v['reference']:.1f} | ECL90 {v['ecl']} | stopped {v['stopped_at']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
