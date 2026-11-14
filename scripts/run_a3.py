"""Run the cells of experiment E0_floor or E1_truncation (configs/a3_floor.yaml). Resumable.

    python scripts/run_a3.py --exp E0_floor --pilot     # pilot_n examples per cell, then the projected runtime
    python scripts/run_a3.py --exp E0_floor             # everything still pending
    python scripts/run_a3.py --exp E1_truncation --methods m0-tail-1.7B

Each reader is loaded once per group of consecutive cells that use it. A lost GPU
stops the run with exit status 75; rerun the same command once the GPU is back.
Pilot mode writes results/<exp>/pilot_projection.json.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.eval import runner as R  # noqa: E402
from pcm.runlog import atomic_write_json  # noqa: E402
from pcm.tasks.common import load_task_config  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "a3_floor.yaml"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_config() -> dict:
    with CONFIG.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cells(cfg: dict, exp: str, only: list[str] | None) -> list[dict]:
    out = []
    for entry in cfg["experiments"][exp]:
        if only and entry["method"] not in only:
            continue
        tasks = entry["tasks"]
        tasks = cfg["task_sets"][tasks] if isinstance(tasks, str) else tasks
        for L in entry["lengths"]:
            for t in tasks:
                out.append({"method": entry["method"], "task": t, "length": L, "indices": entry.get("indices")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, choices=["E0_floor", "E1_truncation"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    tcfg = load_task_config()
    todo = cells(cfg, args.exp, args.methods)
    log(f"{args.exp}: {len(todo)} cells{' (pilot)' if args.pilot else ''}")

    summaries = []
    current = None  # (method name, instance)
    t_start = time.perf_counter()
    for c in todo:
        mcfg = cfg["methods"][c["method"]]
        if current is None or current[0] != c["method"]:
            prev_reader = cfg["methods"][current[0]]["reader"] if current else None
            if prev_reader is not None and prev_reader != mcfg["reader"]:
                R.release_readers()
            inst = R.make_method(mcfg)
            inst.setup()
            current = (c["method"], inst)
        n = None if c["indices"] else cfg["n"]
        s = R.run(mcfg, c["task"], c["length"], n, seed=cfg["seed"], exp=args.exp, pilot=args.pilot, pilot_n=cfg["pilot_n"],
                  retry_failed=args.retry_failed, indices=c["indices"], method=current[1], cfg=tcfg, log=log)
        summaries.append(s)
    elapsed = time.perf_counter() - t_start

    if args.pilot:
        total = sum(s.get("projected_s_remaining") or 0.0 for s in summaries)
        proj = {
            "exp": args.exp, "pilot_elapsed_s": elapsed, "projected_remaining_s": total,
            "projected_remaining_h": total / 3600, "peak_vram_gib": max((s["peak_vram_gib"] or 0) for s in summaries),
            "cells": summaries,
        }
        atomic_write_json(REPO_ROOT / "results" / args.exp / "pilot_projection.json", proj)
        by_method: dict = {}
        for s in summaries:
            by_method[s["method"]] = by_method.get(s["method"], 0.0) + (s.get("projected_s_remaining") or 0.0)
        log(f"pilot done in {elapsed / 60:.1f} min. Projected time for the rest: {total / 3600:.2f} h; "
            f"peak VRAM {proj['peak_vram_gib']:.2f} GiB. By method: " + ", ".join(f"{k} {v / 3600:.2f} h" for k, v in by_method.items()))
    else:
        ok = sum(s["ok"] for s in summaries)
        log(f"done in {elapsed / 3600:.2f} h: {ok} ok, {sum(s['failed'] for s in summaries)} failed, "
            f"{sum(s['skipped'] for s in summaries)} skipped in this session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
