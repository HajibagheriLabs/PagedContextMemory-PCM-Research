"""Run the cells of E2_dense, E3_retrieval or E3o_oracle (configs/a4_baselines.yaml). Resumable.

    python scripts/run_a4.py --exp E3o_oracle --pilot           # pilot only: 5 examples per cell, projected runtime
    python scripts/run_a4.py --exp E3o_oracle --max-hours 14    # pilot where it is missing, then everything pending
    python scripts/run_a4.py --exp E2_dense --max-hours 16      # pilot, N plan under the cap, then the run
    python scripts/run_a4.py --exp E3_retrieval --pilot --tasks synth:S1 babilong:qa1 --lengths 256k

Every invocation first makes sure each cell has its pilot rows (they are ordinary rows and
count towards the cell), then prints the projected time for what is left. A full run whose
projection exceeds --max-hours (default: ask_above_hours of the config) stops there and says so.

E2 (dense YaRN): the YaRN factor of a cell comes from the longest prompt of its split. After the
pilot the number of examples per cell is fixed under the experiment's time cap and written to
results/E2_dense/n_plan.json (only by a pilot over all cells); later invocations read it.

A lost GPU stops the run with exit status 75; rerun the same command once the GPU is back.
Output also goes to logs/A4/<exp>.log.
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

from pcm.eval import runner as R  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.common import config_hash, get_tokenizer, load_task_config  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "a4_baselines.yaml"
EXPERIMENTS = ("E2_dense", "E3_retrieval", "E3o_oracle")
GIB = 1024**3
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


def cells(cfg: dict, exp: str, methods=None, tasks=None, lengths=None) -> list[dict]:
    """The experiment's cells in run order, each {method, task, length, n: "all" | "plan" | "fixed"}."""
    out = []
    for entry in cfg["experiments"][exp]:
        if methods and entry["method"] not in methods:
            continue
        names = entry["tasks"]
        names = cfg["task_sets"][names] if isinstance(names, str) else names
        for L in entry["lengths"]:
            if lengths and L not in lengths:
                continue
            for t in names:
                if tasks and t not in tasks:
                    continue
                out.append({"method": entry["method"], "task": t, "length": L, "n": entry.get("n", "all")})
    return out


# ---------------------------------------------------------------- examples of a cell
def answer_budget(mcfg: dict, example: dict) -> int:
    key = f"{example.get('benchmark', 'synth')}:{example['task']}"
    return int((mcfg.get("max_new_tokens") or {}).get(key, example["max_new_tokens"]))


def cell_examples(cell: dict, n: int, cfg: dict, tcfg: dict) -> list[dict]:
    """The n examples of a cell: the first n (synthetic, BABILong), or n of the task's 100 evenly
    spaced rows, themselves evenly spaced (InfiniteBench), so that a smaller n stays a subset."""
    if cell["task"].startswith("infbench:"):
        base = R.load_examples(cell["task"], cell["length"], cfg["n"], cfg["seed"], tcfg)
        return [base[i] for i in R.even_subset(len(base), n)]
    return R.load_examples(cell["task"], cell["length"], n, cfg["seed"], tcfg)


def split_max_tokens(cell: dict, mcfg: dict, cfg: dict, tcfg: dict) -> int:
    """The longest prompt plus answer budget over all rows of the cell's split, in tokens."""
    bench, name = R.parse_task(cell["task"])
    tok = get_tokenizer(tcfg["tokenizer"])
    override = (mcfg.get("max_new_tokens") or {}).get(cell["task"])
    if bench == "babilong":
        from pcm.tasks import babilong

        meta = babilong.ensure_meta(name, cell["length"], tok, tcfg)
        return max(m["prompt_tokens"] for m in meta.values()) + int(override or tcfg["babilong"]["max_new_tokens"])
    if bench == "infbench":
        from pcm.tasks import infbench
        from pcm.tasks.official import infbench as official

        meta = infbench.ensure_meta(name, tok, tcfg)
        return max(m["prompt_tokens"] for m in meta.values()) + int(override or official().max_new_tokens[name])
    from pcm.models import chat_segments
    from pcm.tasks.common import count_tokens

    longest = 0
    for ex in R.load_examples(cell["task"], cell["length"], tcfg["synth"]["n_per_cell"], cfg["seed"], tcfg):
        chat = chat_segments(tok, *R.prompt_for(ex, tcfg), "")
        longest = max(longest, count_tokens(tok, chat.prefix) + ex["length_tokens"] + count_tokens(tok, chat.suffix) + answer_budget(mcfg, ex))
    return longest


def reader_geometry(reader_key: str) -> dict:
    """Original position range and KV bytes per token of a reader, from its shipped config."""
    from pcm.hub import model_path
    from pcm.memory import kv_bytes_per_token
    from pcm.models import shipped_rope

    path = model_path(reader_key)
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    head_dim = raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"]
    return {"original_positions": shipped_rope(path)["max_position_embeddings"],
            "kv_bytes_per_token": kv_bytes_per_token(raw["num_hidden_layers"], raw["num_key_value_heads"], head_dim)}


def prepare_dense_cells(todo: list[dict], cfg: dict, tcfg: dict) -> None:
    """Give every E2 cell its YaRN factor (section 4.7 rule 5) and the free VRAM it needs."""
    from pcm.models import yarn_factor_for

    geo: dict = {}
    for c in todo:
        base = cfg["methods"][c["method"]]
        g = geo.setdefault(base["reader"], reader_geometry(base["reader"]))
        c["max_tokens"] = split_max_tokens(c, base, cfg, tcfg)
        factor = yarn_factor_for(c["max_tokens"], g["original_positions"], tuple(cfg["e2"]["yarn_factors"]))
        if factor is None:
            raise ValueError(f"{c['task']} {c['length']} fits the model's own range ({c['max_tokens']} tokens): not a dense-extension cell")
        c["mcfg"] = {**base, "yarn_factor": factor}
        c["need_vram_gib"] = c["max_tokens"] * g["kv_bytes_per_token"] / GIB + float(cfg["e2"]["working_reserve_gib"])


# ---------------------------------------------------------------- what the log already holds
def read_rows(exp_dir: Path) -> list[dict]:
    last: dict = {}
    for r in JsonlLog(exp_dir / "runs.jsonl").rows():
        last[r["id"]] = r
    return list(last.values())


def cell_key(c: dict) -> tuple:
    return (config_hash(c["mcfg"]), c["task"], c["length"])


def cell_stats(rows: list[dict]) -> dict:
    """Per (method hash, task, length): rows done, and wall seconds per answered example
    (the longer of the GPU step and the CPU preparation that overlaps it)."""
    stats: dict = defaultdict(lambda: {"done": 0, "ok": 0, "failed": 0, "seconds": []})
    for r in rows:
        s = stats[(r["method_hash"], r["task"], r["length"])]
        s["done"] += 1
        if r["status"] == "ok":
            s["ok"] += 1
            t = r.get("timings") or {}
            s["seconds"].append(max(t.get("total_s", 0.0), t.get("prepare_s", 0.0)))
        elif r["status"] == "failed":
            s["failed"] += 1
    return stats


def project(todo: list[dict], stats: dict) -> dict:
    """Seconds left per cell, from the rows the cell already has."""
    total, per_cell, unmeasured = 0.0, [], []
    for c in todo:
        s = stats.get(cell_key(c))
        if not s or not s["done"]:
            unmeasured.append(f"{c['method']} {c['task']} {c['length']}")
            continue
        sec = sum(s["seconds"]) / len(s["seconds"]) if s["seconds"] else 0.0
        share = s["ok"] / s["done"]  # skipped examples cost almost nothing
        left = max(0, c["n_run"] - s["done"])
        total += sec * left * share
        per_cell.append({"method": c["method"], "task": c["task"], "length": c["length"], "n": c["n_run"], "done": s["done"],
                         "seconds_per_example": round(sec, 3), "remaining_s": round(sec * left * share, 1)})
    return {"remaining_s": total, "remaining_h": total / 3600, "cells": per_cell, "unmeasured": unmeasured}


# ---------------------------------------------------------------- E2: examples per cell under the cap
def make_plan(todo: list[dict], stats: dict, cfg: dict) -> dict:
    """The largest n of the candidates for the `plan` cells that keeps the projected E2 total within the cap."""
    e2 = cfg["e2"]
    sec = {}
    for c in todo:
        s = stats.get(cell_key(c))
        if not s or not s["seconds"]:
            raise RuntimeError(f"no pilot timing for {c['method']} {c['task']} {c['length']}: the plan needs every cell")
        sec[f"{c['task']}|{c['length']}"] = sum(s["seconds"]) / len(s["seconds"])
    fixed_s = sum(e2["n_fixed"] * sec[f"{c['task']}|{c['length']}"] for c in todo if c["n"] == "fixed")
    per_n = sum(sec[f"{c['task']}|{c['length']}"] for c in todo if c["n"] == "plan")
    cap_s = e2["cap_hours"] * 3600
    options = {int(n): (fixed_s + n * per_n) / 3600 for n in e2["n_candidates"]}
    fitting = [n for n in sorted(options, reverse=True) if options[n] * 3600 <= cap_s]
    return {"n_plan": fitting[0] if fitting else None, "n_fixed": int(e2["n_fixed"]), "cap_hours": e2["cap_hours"],
            "projected_hours_by_n": {str(k): round(v, 2) for k, v in options.items()},
            "projected_hours": round(options[fitting[0]], 2) if fitting else None,
            "seconds_per_example": {k: round(v, 2) for k, v in sec.items()}}


def n_for(cell: dict, cfg: dict, plan: dict | None) -> int | None:
    if cell["n"] == "all":
        return int(cfg["n"])
    if cell["n"] == "fixed":
        return int(cfg["e2"]["n_fixed"])
    return int(plan["n_plan"]) if plan and plan.get("n_plan") else None  # `plan` cells before the plan exists: pilot rows only


# ---------------------------------------------------------------- running
class Session:
    """Keeps one reader loaded at a time (a dense 128K run leaves no room for a second one)."""

    def __init__(self):
        self.reader_key = None
        self.current = None

    def method_for(self, mcfg: dict):
        inst = R.make_method(mcfg)
        key = inst.reader_key() if hasattr(inst, "reader_key") else None
        if self.reader_key is not None and key != self.reader_key:
            self.current = None  # the previous method holds the previous reader
            R.release_readers(keep_index_models=True)
        inst.setup()
        self.reader_key = key
        self.current = inst
        return inst


def run_cells(todo: list[dict], args, cfg: dict, tcfg: dict, stats: dict, session: Session, pilot: bool) -> list[dict]:
    from pcm.health import preflight

    import gc

    import torch

    summaries = []
    for c in todo:
        s = stats.get(cell_key(c), {"done": 0, "failed": 0})
        done = s["done"] - (s["failed"] if args.retry_failed else 0)
        target = cfg["pilot_n"] if pilot else c["n_run"]
        if target is None or done >= target:
            continue
        examples = cell_examples(c, cfg["pilot_n"] if c["n_run"] is None else c["n_run"], cfg, tcfg)
        inst = session.method_for(c["mcfg"])
        if "need_vram_gib" in c:
            # Memory the previous cell's cache left reserved in this process is free for this one: return it to the
            # driver first, so that the check measures what the cell can really use.
            gc.collect()
            torch.cuda.empty_cache()
            pf = preflight(min_free_vram_gib=c["need_vram_gib"])
            log(f"dense cell: longest prompt {c['max_tokens']} tokens, YaRN factor {c['mcfg']['yarn_factor']}, "
                f"needs {c['need_vram_gib']:.1f} GiB free VRAM, has {pf['free_vram_gib']:.1f}")
        summaries.append(R.run(c["mcfg"], c["task"], c["length"], None, seed=cfg["seed"], exp=args.exp, pilot=pilot,
                               pilot_n=max(0, cfg["pilot_n"] - done) if pilot else cfg["pilot_n"], retry_failed=args.retry_failed,
                               examples=examples, method=inst, cfg=tcfg, log=log))
        # Drop this cell's method: holding it would keep its reader in memory while the next cell loads another one.
        inst = examples = None
        session.current = None
    return summaries


def main() -> int:
    global _LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=EXPERIMENTS)
    ap.add_argument("--pilot", action="store_true", help="stop after the pilot rows and the projection")
    ap.add_argument("--max-hours", type=float, default=None, help="largest projected run this invocation may start")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    cfg, tcfg = load_config(), load_task_config()
    exp_dir = REPO_ROOT / "results" / args.exp
    _LOG_FILE = REPO_ROOT / "logs" / "A4" / f"{args.exp}.log"
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered = bool(args.methods or args.tasks or args.lengths)
    todo = cells(cfg, args.exp, args.methods, args.tasks, args.lengths)
    if not todo:
        log("no cell matches the filters")
        return 1
    dense = args.exp == "E2_dense"
    if dense:
        prepare_dense_cells(todo, cfg, tcfg)
    else:
        for c in todo:
            c["mcfg"] = cfg["methods"][c["method"]]
    plan_path = exp_dir / "n_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    for c in todo:
        c["n_run"] = n_for(c, cfg, plan)
    log(f"{args.exp}: {len(todo)} cells{' (filtered)' if filtered else ''}{' (pilot only)' if args.pilot else ''}")

    session = Session()
    t_start = time.perf_counter()
    # ---- pilot rows for every cell that lacks them
    run_cells(todo, args, cfg, tcfg, cell_stats(read_rows(exp_dir)), session, pilot=True)
    stats = cell_stats(read_rows(exp_dir))
    if dense and plan is None:
        if filtered:
            log("the N plan is made by a pilot over all cells: run without filters to fix it")
        else:
            plan = make_plan(todo, stats, cfg)
            if plan["n_plan"] is None:
                atomic_write_json(exp_dir / "n_plan_over_cap.json", plan)
                log(f"even {min(cfg['e2']['n_candidates'])} examples per cell exceed the cap of {plan['cap_hours']} h "
                    f"({plan['projected_hours_by_n']}). Not going below it: the owner decides.")
                return 2
            atomic_write_json(plan_path, plan)
            log(f"N plan: {plan['n_plan']} examples in the cells of the H1 families, {plan['n_fixed']} in the others; "
                f"projected {plan['projected_hours']} h of {plan['cap_hours']} h (by n: {plan['projected_hours_by_n']})")
            for c in todo:
                c["n_run"] = n_for(c, cfg, plan)
    runnable = [c for c in todo if c["n_run"] is not None]
    proj = project(runnable, stats)
    by_method: Counter = Counter()
    for pc in proj["cells"]:
        by_method[pc["method"]] += pc["remaining_s"]
    proj.update({"exp": args.exp, "filtered": filtered, "pilot_elapsed_s": round(time.perf_counter() - t_start, 1),
                 "by_method_h": {k: round(v / 3600, 2) for k, v in by_method.items()}})
    atomic_write_json(exp_dir / ("pilot_projection_partial.json" if filtered else "pilot_projection.json"), proj)
    log(f"projected time for what is left: {proj['remaining_h']:.2f} h. By method: "
        + ", ".join(f"{k} {v} h" for k, v in proj["by_method_h"].items())
        + (f". No timing yet for {len(proj['unmeasured'])} cells." if proj["unmeasured"] else ""))
    if args.pilot:
        return 0
    limit = args.max_hours if args.max_hours is not None else float(cfg["ask_above_hours"])
    if proj["remaining_h"] > limit:
        log(f"the run is projected at {proj['remaining_h']:.1f} h, above {limit:g} h: start it with --max-hours "
            f"{int(proj['remaining_h']) + 2} (or larger) to go ahead.")
        return 3

    # ---- everything pending
    summaries = run_cells(runnable, args, cfg, tcfg, stats, session, pilot=False)
    elapsed = time.perf_counter() - t_start
    log(f"done in {elapsed / 3600:.2f} h: {sum(s['ok'] for s in summaries)} ok, {sum(s['failed'] for s in summaries)} failed, "
        f"{sum(s['skipped'] for s in summaries)} skipped in this session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
