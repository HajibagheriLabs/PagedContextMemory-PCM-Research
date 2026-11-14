"""Tables and curves from results/<exp>/runs.jsonl.

    python scripts/report.py --exp E0_floor
    python scripts/report.py --exp E3_retrieval --reference-exp E0_floor --reference-method native-1.7B

Reads every row, keeps the last row per id (a retried failure supersedes the
failed row), and writes to results/<exp>/:
  report_tables.md   accuracy with 95% bootstrap CI per method, task and length; status
                     counts; evidence recall; seconds per example; peak VRAM; ECL90 when
                     a reference is given; accuracy by depth decile; the code commits seen
  report_summary.json the same numbers, machine-readable
  figures/acc_<task>.png accuracy-versus-length curves with CI bars

ECL90 reference: the accuracy of --reference-method at the reference length
(configs/a2_tasks.yaml, metrics.reference_length) on the same task, read from
--reference-exp (default: the same experiment).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.eval import metrics  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json, atomic_write_text  # noqa: E402
from pcm.tasks.common import load_task_config  # noqa: E402

LENGTH_ORDER = ["0k", "1k", "2k", "4k", "8k", "16k", "32k", "64k", "128k", "256k", "512k", "1M", "full"]


def latest_rows(exp_dir: Path) -> tuple[list[dict], int]:
    log = JsonlLog(exp_dir / "runs.jsonl")
    last: dict[str, dict] = {}
    for row in log.rows():
        last[row["id"]] = row
    return list(last.values()), log.bad_lines


def chance_of(task: str, cfg: dict) -> float:
    bench, name = task.split(":", 1)
    if bench == "synth":
        return float(cfg["synth"]["tasks"][name]["chance"])
    return float(cfg[bench]["chance"][name])


def length_key(L: str) -> int:
    return LENGTH_ORDER.index(L) if L in LENGTH_ORDER else len(LENGTH_ORDER)


def cell_stats(rows: list[dict], mcfg: dict) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    ci = metrics.bootstrap_ci([r["score"] for r in ok], mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"])
    rec = [r["evidence_recall"] for r in ok if r.get("evidence_recall") is not None]
    secs = [r["timings"]["total_s"] for r in ok if r.get("timings")]
    vram = [r["peak_vram_gib"] for r in rows if r.get("peak_vram_gib") is not None]
    return {
        "accuracy": ci,
        "status": {s: sum(1 for r in rows if r.get("status") == s) for s in ("ok", "failed", "skipped")},
        "evidence_recall": sum(rec) / len(rec) if rec else None,
        "seconds_per_example": sum(secs) / len(secs) if secs else None,
        "peak_vram_gib": max(vram) if vram else None,
        "tokens_mean": sum(r["length_tokens"] for r in ok) / len(ok) if ok and ok[0].get("length_tokens") is not None else None,
    }


def fmt_acc(c: dict) -> str:
    if c["mean"] is None:
        return "–"
    return f"{100 * c['mean']:.1f} [{100 * c['lo']:.1f}, {100 * c['hi']:.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--reference-exp", default=None)
    ap.add_argument("--reference-method", default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--results-root", default=str(REPO_ROOT / "results"), help="directory holding <exp>/ (default: results/)")
    args = ap.parse_args()
    root = Path(args.results_root)
    cfg = load_task_config()
    mcfg = cfg["metrics"]
    exp_dir = root / args.exp
    rows, bad = latest_rows(exp_dir)
    if not rows:
        print(f"no rows in {exp_dir / 'runs.jsonl'}")
        return 1

    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["method"], r["task"], r["length"])].append(r)
    stats = {k: cell_stats(v, mcfg) for k, v in cells.items()}

    commits = sorted({(r.get("git_commit") or "none")[:10] + ("+dirty" if r.get("git_dirty") else "") for r in rows})
    lines = [f"# {args.exp}: tables from runs.jsonl", "",
             f"Rows: {len(rows)} (last row per id); malformed lines skipped: {bad}.",
             f"Code commits of these rows: {', '.join(commits)}" + ("  **(more than one: check before reporting)**" if len(commits) > 1 else ""), ""]
    lines += ["## Accuracy (%, 95% bootstrap CI)", "", "| method | task | length | n ok | failed | skipped | accuracy | evidence recall | s/example | peak VRAM GiB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, t, L) in sorted(stats, key=lambda k: (k[0], k[1], length_key(k[2]))):
        s = stats[(m, t, L)]
        er = f"{s['evidence_recall']:.3f}" if s["evidence_recall"] is not None else "–"
        sp = f"{s['seconds_per_example']:.2f}" if s["seconds_per_example"] is not None else "–"
        pv = f"{s['peak_vram_gib']:.2f}" if s["peak_vram_gib"] is not None else "–"
        lines.append(f"| {m} | {t} | {L} | {s['status']['ok']} | {s['status']['failed']} | {s['status']['skipped']} | {fmt_acc(s['accuracy'])} | {er} | {sp} | {pv} |")

    ecl_out = {}
    if args.reference_method:
        ref_rows, _ = latest_rows(root / (args.reference_exp or args.exp))
        ref_len = mcfg["reference_length"]
        grid = list(cfg["lengths"])
        lines += ["", f"## ECL90 (reference: {args.reference_method} at {ref_len})", "", "| method | task | reference | threshold | floor | ECL90 | stopped at |", "|---|---|---|---|---|---|---|"]
        for t in sorted({k[1] for k in stats}):
            ref = [r for r in ref_rows if r["method"] == args.reference_method and r["task"] == t and r["length"] == ref_len and r.get("status") == "ok"]
            if not ref:
                continue
            ref_acc = sum(r["score"] for r in ref) / len(ref)
            for m in sorted({k[0] for k in stats if k[1] == t}):
                curve = {L: stats[(m, t, L)]["accuracy"]["mean"] for (mm, tt, L) in stats if mm == m and tt == t}
                e = metrics.ecl90(curve, ref_acc, chance_of(t, cfg), grid, mcfg["ecl_ratio"], mcfg["ecl_floor_min"])
                ecl_out[f"{m}|{t}"] = e
                st = e["stopped_at"]
                lines.append(f"| {m} | {t} | {100 * ref_acc:.1f} | {100 * e['threshold']:.1f} | {100 * e['floor']:.1f} | {e['ecl'] or 'none'} | {st['length'] + ' (' + st['reason'] + ')' if st else 'end of grid'} |")

    lines += ["", "## Accuracy by evidence depth decile (%)", ""]
    dec_out = {}
    for (m, t, L) in sorted(stats, key=lambda k: (k[0], k[1], length_key(k[2]))):
        ok = [r for r in cells[(m, t, L)] if r.get("status") == "ok"]
        by = metrics.accuracy_by_decile(ok, mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"])
        if not by:
            continue
        dec_out[f"{m}|{t}|{L}"] = by
        lines.append(f"- {m} {t} {L}: " + ", ".join(f"d{d} {100 * c['mean']:.0f} (n={c['n']})" for d, c in by.items()))

    atomic_write_text(exp_dir / "report_tables.md", "\n".join(lines) + "\n")
    atomic_write_json(exp_dir / "report_summary.json", {
        "commits": commits, "rows": len(rows), "bad_lines": bad,
        "cells": {f"{m}|{t}|{L}": s for (m, t, L), s in stats.items()}, "ecl90": ecl_out, "deciles": dec_out,
    })

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = exp_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for t in sorted({k[1] for k in stats}):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for m in sorted({k[0] for k in stats if k[1] == t}):
                Ls = sorted([k[2] for k in stats if k[0] == m and k[1] == t], key=length_key)
                acc = [stats[(m, t, L)]["accuracy"] for L in Ls]
                xs = [i for i, a in enumerate(acc) if a["mean"] is not None]
                if not xs:
                    continue
                y = [100 * acc[i]["mean"] for i in xs]
                err = [[100 * (acc[i]["mean"] - acc[i]["lo"]) for i in xs], [100 * (acc[i]["hi"] - acc[i]["mean"]) for i in xs]]
                ax.errorbar([Ls[i] for i in xs], y, yerr=err, marker="o", capsize=3, label=m)
            ax.set_ylim(0, 100)
            ax.set_ylabel("accuracy (%)")
            ax.set_xlabel("context length")
            ax.set_title(t)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / f"acc_{t.replace(':', '_')}.png", dpi=120)
            plt.close(fig)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
