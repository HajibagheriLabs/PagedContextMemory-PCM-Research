"""Run A5 (configs/a5_pillars.yaml): E4 pages x budget, then E5 graph ablation. Resumable.

    python scripts/run_a5.py --stage screen --pilot          # E4 screen: pilot rows of every cell, then the projection
    python scripts/run_a5.py --stage screen --max-hours 8    # E4 screen: 27 settings x 3 tasks x 128K/256K, N = 30
    python scripts/run_a5.py --stage final --max-hours 12    # E4 final: the best three settings on every task, N = 100
    python scripts/run_a5.py --stage e5 --max-hours 30       # E5: graph variants at E4's best setting
    python scripts/run_a5.py --stage all --max-hours 50      # the three stages in order

Every stage first gives each of its cells its pilot rows (ordinary rows that count towards the cell), then
prints the projected time for what is left, and goes on only if that fits --max-hours (with --stage all,
the hours left of the budget). The choices between stages follow rules fixed in the config before any
run: the final stage needs the complete screen and writes results/E4_pages/best_settings.json; E5 needs
the complete final stage and writes results/E4_pages/e5_setting.json. Neither file is rewritten.

E5's spaCy variants need the entities of their documents: scripts/a5_ingest_entities.py computes them on
the CPU beforehand (otherwise each document costs about 35 s of CPU inside the run).

A lost GPU stops the run with exit status 75; rerun the same command once the GPU is back.
Output also goes to logs/A5/<stage>.log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcm.eval import runner as R  # noqa: E402
from pcm.runlog import atomic_write_json  # noqa: E402
from pcm.tasks.common import config_hash, load_task_config  # noqa: E402
from run_a4 import Session, cell_examples, cell_key, cell_stats, read_rows  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "a5_pillars.yaml"
STAGES = ("screen", "final", "e5")
EXP = {"screen": "E4_pages", "final": "E4_pages", "e5": "E5_graph"}
_LOG_FILE = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d ") + line + "\n")


def load_config(path: Path = CONFIG) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------- method configurations
def setting_config(cfg: dict, page: int, budget: int, variant: str = "none", selection_only: bool = False) -> dict:
    """The method configuration of one setting (and graph variant). The no-graph setting has no `graph` key, so
    it is the same configuration, and the same hash, in E4 and E5."""
    m = {**cfg["pcm"], "page": int(page), "budget": int(budget)}
    name = f"pcm-p{page}-b{budget}"
    spec = cfg["e5"]["variants"].get(variant) if variant != "none" else None
    if variant != "none":
        if variant not in cfg["e5"]["variants"]:
            raise ValueError(f"unknown graph variant {variant!r}")
        m["graph"] = {**cfg["graph"], "extractor": "default", **spec}
        name += f"-{variant}"
    if selection_only:
        m["selection_only"] = True
        m["min_free_vram_gib"] = 4.0  # no reader: the embedder alone
        name += "-sel"
    m["name"] = name
    return m


def screen_settings(cfg: dict) -> list[tuple[int, int]]:
    """(page, budget) pairs of the E4 grid in which at least one page fits the budget."""
    return [(p, b) for b in cfg["e4"]["budgets"] for p in cfg["e4"]["pages"] if p <= b]


def expand_tasks(cfg: dict, names) -> list[str]:
    out = []
    for n in names:
        out += cfg["task_sets"][n] if n in cfg["task_sets"] else [n]
    return out


def lengths_for(task: str, lengths) -> list[str]:
    return ["full"] if task.startswith("infbench:") else list(lengths)


def cells_for(cfg: dict, stage: str, chosen: dict | None) -> list[dict]:
    """The stage's cells in run order: {method, mcfg, task, length, n}. Selection-only cells last (no reader)."""
    out, sel = [], []

    def add(mcfg, task, L, n, selection_only=False):
        (sel if selection_only else out).append({"method": mcfg["name"], "mcfg": mcfg, "task": task, "length": L, "n": n})

    if stage == "screen":
        sc = cfg["e4"]["screen"]
        for p, b in screen_settings(cfg):
            for L in sc["lengths"]:
                for t in sc["tasks"]:
                    add(setting_config(cfg, p, b), t, L, sc["n"])
                for t in sc["recall_only"]:
                    add(setting_config(cfg, p, b, selection_only=True), t, L, sc["n"], True)
    elif stage == "final":
        fin = cfg["e4"]["final"]
        tasks = expand_tasks(cfg, fin["tasks"])
        for s in chosen["settings"]:
            seen = set()
            for L in fin["lengths"]:
                for t in tasks:
                    for LL in lengths_for(t, [L]):
                        if (t, LL) not in seen:
                            seen.add((t, LL))
                            add(setting_config(cfg, s["page"], s["budget"]), t, LL, cfg["n"])
    else:
        p, b = chosen["page"], chosen["budget"]
        for entry in cfg["e5"]["plan"]:
            tasks = expand_tasks(cfg, entry["tasks"])
            for v in entry["variants"]:
                seen = set()
                for L in entry["lengths"]:
                    for t in tasks:
                        for LL in lengths_for(t, [L]):
                            if (t, LL) not in seen:
                                seen.add((t, LL))
                                so = bool(entry.get("selection_only"))
                                add(setting_config(cfg, p, b, v, selection_only=so), t, LL, cfg["n"], so)
    # One cell per (configuration, task, length), first occurrence kept.
    seen, uniq = set(), []
    for c in out + sel:
        k = (c["method"], c["task"], c["length"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


# ---------------------------------------------------------------- choices between stages (rules fixed in the config)
def _acc(rows):
    s = [r["score"] for r in rows if r["status"] == "ok" and r.get("score") is not None]
    return sum(s) / len(s) if s else None


def _recall(rows):
    s = [r["evidence_recall"] for r in rows if r["status"] == "ok" and r.get("evidence_recall") is not None]
    return sum(s) / len(s) if s else None


def rows_by_cell(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["method_hash"], r["task"], r["length"])].append(r)
    return by


def choose_best_settings(cfg: dict, rows: list[dict]) -> dict:
    """The best `top` settings of the screen: mean accuracy over the accuracy cells (equal weights), then mean
    evidence recall over the same cells, then the smaller budget, then the smaller page. Needs every cell complete."""
    sc = cfg["e4"]["screen"]
    by = rows_by_cell(rows)
    table = []
    for p, b in screen_settings(cfg):
        h = config_hash(setting_config(cfg, p, b))
        accs, recs, incomplete = [], [], []
        for L in sc["lengths"]:
            for t in sc["tasks"]:
                rs = [r for r in by.get((h, t, L), []) if r["status"] == "ok"]
                if len(rs) < sc["n"]:
                    incomplete.append(f"{t} {L} ({len(rs)}/{sc['n']})")
                accs.append(_acc(rs))
                recs.append(_recall(rs))
        if incomplete:
            raise RuntimeError(f"screen incomplete for page {p}, budget {b}: {', '.join(incomplete)}")
        table.append({"page": p, "budget": b, "accuracy": sum(accs) / len(accs), "evidence_recall": sum(recs) / len(recs),
                      "cells": {f"{t}|{L}": {"accuracy": a, "evidence_recall": rc} for (t, L), a, rc in
                                zip([(t, L) for L in sc["lengths"] for t in sc["tasks"]], accs, recs)}})
    table.sort(key=lambda s: (-s["accuracy"], -s["evidence_recall"], s["budget"], s["page"]))
    top = cfg["e4"]["final"]["top"]
    return {"rule": "mean accuracy over the screen cells; ties: mean evidence recall, smaller budget, smaller page",
            "settings": [{"page": s["page"], "budget": s["budget"]} for s in table[:top]], "ranking": table}


def choose_e5_setting(cfg: dict, rows: list[dict], best: dict) -> dict:
    """E5's setting: of the final settings, the highest mean over 128K and 256K of the H1 family accuracies (S1, S2,
    BABILong; a family's accuracy is the mean of its tasks'); ties: smaller budget, smaller page."""
    by = rows_by_cell(rows)
    fams = {f: cfg["families"][f] for f in cfg["h1_families"]}
    table = []
    for s in best["settings"]:
        h = config_hash(setting_config(cfg, s["page"], s["budget"]))
        per_len = {}
        for L in cfg["e4"]["final"]["lengths"]:
            fam_acc = {}
            for f, tasks in fams.items():
                accs = []
                for t in tasks:
                    rs = [r for r in by.get((h, t, L), []) if r["status"] == "ok"]
                    if len(rs) < cfg["n"]:
                        raise RuntimeError(f"final stage incomplete: page {s['page']}, budget {s['budget']}, {t} {L} ({len(rs)}/{cfg['n']})")
                    accs.append(_acc(rs))
                fam_acc[f] = sum(accs) / len(accs)
            per_len[L] = fam_acc
        score = sum(sum(v.values()) / len(v) for v in per_len.values()) / len(per_len)
        table.append({"page": s["page"], "budget": s["budget"], "score": score, "families": per_len})
    table.sort(key=lambda s: (-s["score"], s["budget"], s["page"]))
    return {"rule": "mean over 128K and 256K of the S1, S2 and BABILong accuracies; ties: smaller budget, smaller page",
            "page": table[0]["page"], "budget": table[0]["budget"], "ranking": table}


# ---------------------------------------------------------------- projection
def project(todo: list[dict], stats: dict) -> dict:
    """Seconds left per cell from its own rows; a cell without rows yet takes the mean of measured cells of the same
    configuration and length class (rare: every cell is piloted first)."""
    total, per_cell, unmeasured = 0.0, [], []
    for c in todo:
        s = stats.get(cell_key(c))
        if not s or not s["done"]:
            unmeasured.append(f"{c['method']} {c['task']} {c['length']}")
            continue
        sec = sum(s["seconds"]) / len(s["seconds"]) if s["seconds"] else 0.0
        share = s["ok"] / s["done"]
        left = max(0, c["n"] - s["done"])
        total += sec * left * share
        per_cell.append({"method": c["method"], "task": c["task"], "length": c["length"], "n": c["n"], "done": s["done"],
                         "seconds_per_example": round(sec, 3), "remaining_s": round(sec * left * share, 1)})
    return {"remaining_s": total, "remaining_h": total / 3600, "cells": per_cell, "unmeasured": unmeasured}


# ---------------------------------------------------------------- running
def run_cells(todo, exp, cfg, tcfg, stats, session, pilot, answer_from=None, retry_failed=False):
    summaries = []
    for c in todo:
        s = stats.get(cell_key(c), {"done": 0, "failed": 0})
        done = s["done"] - (s["failed"] if retry_failed else 0)
        target = cfg["pilot_n"] if pilot else c["n"]
        if done >= target:
            continue
        examples = cell_examples(c, c["n"], cfg, tcfg)
        inst = session.method_for(c["mcfg"])
        summaries.append(R.run(c["mcfg"], c["task"], c["length"], None, seed=cfg["seed"], exp=exp, pilot=pilot,
                               pilot_n=max(0, cfg["pilot_n"] - done) if pilot else cfg["pilot_n"], retry_failed=retry_failed,
                               examples=examples, method=inst, cfg=tcfg, log=log, answer_from=answer_from))
        inst = examples = None
        session.current = None
    return summaries


def run_stage(stage: str, args, cfg: dict, tcfg: dict, hours_left: float | None) -> tuple[int, float]:
    """Pilot, project, run one stage. Returns (exit status, hours spent)."""
    exp = EXP[stage]
    exp_dir = REPO_ROOT / "results" / exp
    t0 = time.perf_counter()
    chosen = None
    if stage == "final":
        path = exp_dir / "best_settings.json"
        if not path.is_file():
            try:
                best = choose_best_settings(cfg, read_rows(exp_dir))
            except RuntimeError as e:
                log(f"final stage: {e}. Run the screen to the end first.")
                return 4, 0.0
            atomic_write_json(path, best)
            log("best settings of the screen: " + ", ".join(f"page {s['page']} budget {s['budget']}" for s in best["settings"]))
        chosen = json.loads(path.read_text(encoding="utf-8"))
    elif stage == "e5":
        e4_dir = REPO_ROOT / "results" / "E4_pages"
        path = e4_dir / "e5_setting.json"
        if not path.is_file():
            best_path = e4_dir / "best_settings.json"
            if not best_path.is_file():
                log("E5 needs the E4 final stage: run --stage screen, then --stage final")
                return 4, 0.0
            try:
                setting = choose_e5_setting(cfg, read_rows(e4_dir), json.loads(best_path.read_text(encoding="utf-8")))
            except RuntimeError as e:
                log(f"E5: {e}. Run the final stage to the end first.")
                return 4, 0.0
            atomic_write_json(path, setting)
            log(f"E5 setting: page {setting['page']}, budget {setting['budget']}")
        chosen = json.loads(path.read_text(encoding="utf-8"))
    todo = cells_for(cfg, stage, chosen)
    if args.tasks:
        todo = [c for c in todo if c["task"] in args.tasks]
    if args.lengths:
        todo = [c for c in todo if c["length"] in args.lengths]
    if args.methods:
        todo = [c for c in todo if c["method"] in args.methods]
    filtered = bool(args.tasks or args.lengths or args.methods)
    answer_from = cfg["e5"].get("answer_from") if stage == "e5" else None
    log(f"stage {stage} ({exp}): {len(todo)} cells{' (filtered)' if filtered else ''}{' (pilot only)' if args.pilot else ''}")

    session = Session()
    run_cells(todo, exp, cfg, tcfg, cell_stats(read_rows(exp_dir)), session, pilot=True, answer_from=answer_from)
    stats = cell_stats(read_rows(exp_dir))
    proj = project(todo, stats)
    by_method: Counter = Counter()
    for pc in proj["cells"]:
        by_method[pc["method"]] += pc["remaining_s"]
    proj.update({"stage": stage, "exp": exp, "filtered": filtered, "pilot_elapsed_s": round(time.perf_counter() - t0, 1),
                 "by_method_h": {k: round(v / 3600, 2) for k, v in sorted(by_method.items())}})
    atomic_write_json(exp_dir / (f"projection_{stage}_partial.json" if filtered else f"projection_{stage}.json"), proj)
    log(f"stage {stage}: projected {proj['remaining_h']:.2f} h for what is left"
        + (f"; no timing yet for {len(proj['unmeasured'])} cells" if proj["unmeasured"] else ""))
    if args.pilot:
        return 0, (time.perf_counter() - t0) / 3600
    limit = hours_left if hours_left is not None else float(cfg["ask_above_hours"])
    if proj["remaining_h"] > limit:
        log(f"stage {stage} is projected at {proj['remaining_h']:.1f} h, above the {limit:.1f} h allowed: rerun with a larger --max-hours to go ahead")
        return 3, (time.perf_counter() - t0) / 3600
    summaries = run_cells(todo, exp, cfg, tcfg, stats, session, pilot=False, answer_from=answer_from, retry_failed=args.retry_failed)
    spent = (time.perf_counter() - t0) / 3600
    log(f"stage {stage} done in {spent:.2f} h: {sum(s['ok'] for s in summaries)} ok, {sum(s['failed'] for s in summaries)} failed, "
        f"{sum(s['skipped'] for s in summaries)} skipped in this session")
    return 0, spent


def main() -> int:
    global _LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--pilot", action="store_true", help="stop after the pilot rows and the projection")
    ap.add_argument("--max-hours", type=float, default=None, help="largest projected run this invocation may start")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    cfg, tcfg = load_config(), load_task_config()
    stages = STAGES if args.stage == "all" else (args.stage,)
    left = args.max_hours
    for st in stages:
        _LOG_FILE = REPO_ROOT / "logs" / "A5" / f"{st}.log"
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        status, spent = run_stage(st, args, cfg, tcfg, left)
        if status:
            return status
        if left is not None:
            left -= spent
    return 0


if __name__ == "__main__":
    sys.exit(main())
