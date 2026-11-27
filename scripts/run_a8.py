"""Run A8 (configs/a8_local.yaml): E8, the integrated local comparison, and the 256K store check. Resumable.

    python scripts/run_a8.py --stage e8 --pilot              # 5 examples per cell, projected time for the rest
    python scripts/run_a8.py --stage e8 --max-hours 12       # pilot where missing, then every pending example
    python scripts/run_a8.py --stage store                   # the whole system at 256K on one card, 3 documents
    python scripts/run_a8.py --stage all --max-hours 12      # e8, then store

E8 adds what earlier steps did not measure (D-052): P-KV (configs/pcm_kv.yaml) on every retained task and length, and
N = 300 at 256K on the synthetic tasks for P-T, P-KV and M2 hybrid. The other series of the comparison are read from the
earlier experiments by scripts/a8_summary.py. The synthetic examples 100-299 at 256K must exist first:

    python scripts/make_data.py --synth --tasks S1 S2_multikey S2_multivalue --lengths 256k --n 300

A lost GPU stops the run with exit status 75; rerun the same command once the GPU is back.
Output also goes to logs/A8/<stage>.log.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcm.eval import runner as R  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.common import config_hash, load_task_config  # noqa: E402
from run_a4 import Session, cell_examples, read_rows  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "a8_local.yaml"
STAGES = ("e8", "store")
EXP = "E8_local"
GIB = 1024**3
_LOG_FILE = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d ") + line + "\n")


def method_from(spec: dict) -> dict:
    """The method configuration named by {file, key}; key is a name or a path of names inside the file."""
    data = yaml.safe_load((REPO_ROOT / spec["file"]).read_text(encoding="utf-8"))
    for k in ([spec["key"]] if isinstance(spec["key"], str) else spec["key"]):
        data = data[k]
    return dict(data)


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_methods"] = {name: method_from(spec) for name, spec in cfg["methods"].items()}
    return cfg


def expand_tasks(cfg: dict, names) -> list[str]:
    out = []
    for n in names:
        out += cfg["task_sets"][n] if n in cfg["task_sets"] else [n]
    return out


# ---------------------------------------------------------------- cells
def e8_cells(cfg: dict) -> list[dict]:
    """E8's cells in run order: {label, method, mcfg, task, length, lo, hi, n_run, ranged, answer_from}."""
    out, seen = [], set()
    for entry in cfg["cells"]:
        m = cfg["_methods"][entry["method"]]
        lo, hi = entry.get("range", [0, int(cfg["n"])])
        for L in entry["lengths"]:
            for t in expand_tasks(cfg, entry["tasks"]):
                key = (entry["method"], t, L, lo, hi)
                if key in seen:
                    raise ValueError(f"cell listed twice: {key}")
                seen.add(key)
                out.append({"label": entry["method"], "method": m["name"], "mcfg": m, "task": t, "length": L,
                            "lo": int(lo), "hi": int(hi), "n_run": int(hi) - int(lo), "ranged": "range" in entry,
                            "answer_from": entry.get("answer_from")})
    return out


def cell_rows(c: dict, by_cell: dict) -> list[dict]:
    rs = by_cell.get((config_hash(c["mcfg"]), c["task"], c["length"]), [])
    if c["ranged"]:
        rs = [r for r in rs if r.get("index") is not None and c["lo"] <= int(r["index"]) < c["hi"]]
    return rs


def cell_stats(c: dict, by_cell: dict) -> dict:
    """Rows done, and wall seconds per answered example (the longer of the GPU step and the CPU preparation)."""
    s = {"done": 0, "ok": 0, "failed": 0, "seconds": [], "reused": 0}
    for r in cell_rows(c, by_cell):
        s["done"] += 1
        if r["status"] == "ok":
            s["ok"] += 1
            t = r.get("timings") or {}
            s["seconds"].append(max(t.get("total_s", 0.0), t.get("prepare_s", 0.0)))
            s["reused"] += bool(t.get("reused"))
        elif r["status"] == "failed":
            s["failed"] += 1
    return s


def rows_by_cell(exp_dir: Path) -> dict:
    out: dict = defaultdict(list)
    for r in read_rows(exp_dir):
        out[(r["method_hash"], r["task"], r["length"])].append(r)
    return out


def project(todo: list[dict], by_cell: dict) -> dict:
    total, per_cell, unmeasured = 0.0, [], []
    for c in todo:
        s = cell_stats(c, by_cell)
        if not s["done"]:
            unmeasured.append(f"{c['label']} {c['task']} {c['length']} [{c['lo']}, {c['hi']})")
            continue
        sec = sum(s["seconds"]) / len(s["seconds"]) if s["seconds"] else 0.0
        left = max(0, c["n_run"] - s["done"])
        rem = sec * left * (s["ok"] / s["done"])
        total += rem
        per_cell.append({"label": c["label"], "method": c["method"], "task": c["task"], "length": c["length"], "range": [c["lo"], c["hi"]],
                         "done": s["done"], "reused": s["reused"], "seconds_per_example": round(sec, 3), "remaining_s": round(rem, 1)})
    return {"remaining_s": total, "remaining_h": total / 3600, "cells": per_cell, "unmeasured": unmeasured}


def run_cells(todo, cfg, tcfg, by_cell, session, pilot, retry_failed=False) -> list[dict]:
    summaries = []
    for c in todo:
        s = cell_stats(c, by_cell)
        done = s["done"] - (s["failed"] if retry_failed else 0)
        if done >= (cfg["pilot_n"] if pilot else c["n_run"]):
            continue
        examples = cell_examples(c, c["hi"], cfg, tcfg)[c["lo"]:c["hi"]]
        if len(examples) != c["n_run"]:
            raise RuntimeError(f"{c['task']} {c['length']}: {len(examples)} examples for [{c['lo']}, {c['hi']})")
        inst = session.method_for(c["mcfg"])
        summaries.append(R.run(c["mcfg"], c["task"], c["length"], None, seed=cfg["seed"], exp=EXP, pilot=pilot,
                               pilot_n=max(0, cfg["pilot_n"] - done) if pilot else cfg["pilot_n"], retry_failed=retry_failed,
                               examples=examples, method=inst, cfg=tcfg, log=log, answer_from=c["answer_from"]))
        inst = examples = None
        session.current = None
    return summaries


def e8_stage(args, cfg: dict, tcfg: dict, hours_left: float | None) -> tuple[int, float]:
    t0 = time.perf_counter()
    todo = e8_cells(cfg)
    if args.tasks:
        todo = [c for c in todo if c["task"] in args.tasks]
    if args.lengths:
        todo = [c for c in todo if c["length"] in args.lengths]
    if args.methods:
        todo = [c for c in todo if c["label"] in args.methods or c["method"] in args.methods]
    exp_dir = REPO_ROOT / "results" / EXP
    log(f"stage e8 ({EXP}): {len(todo)} cells{' (pilot only)' if args.pilot else ''}")
    session = Session()
    run_cells(todo, cfg, tcfg, rows_by_cell(exp_dir), session, True)
    by_cell = rows_by_cell(exp_dir)
    proj = project(todo, by_cell)
    by_method: Counter = Counter()
    for pc in proj["cells"]:
        by_method[pc["label"]] += pc["remaining_s"]
    proj.update({"stage": "e8", "by_method_h": {k: round(v / 3600, 2) for k, v in sorted(by_method.items())},
                 "pilot_elapsed_s": round(time.perf_counter() - t0, 1)})
    filtered = bool(args.tasks or args.lengths or args.methods)
    atomic_write_json(exp_dir / ("projection_e8_partial.json" if filtered else "projection_e8.json"), proj)
    log(f"stage e8: projected {proj['remaining_h']:.2f} h for what is left (" + ", ".join(f"{k} {v} h" for k, v in proj["by_method_h"].items()) + ")")
    if args.pilot:
        return 0, (time.perf_counter() - t0) / 3600
    limit = hours_left if hours_left is not None else float(cfg["ask_above_hours"])
    if proj["remaining_h"] > limit:
        log(f"stage e8 is projected at {proj['remaining_h']:.1f} h, above the {limit:.1f} h allowed: rerun with a larger --max-hours")
        return 3, (time.perf_counter() - t0) / 3600
    summaries = run_cells(todo, cfg, tcfg, by_cell, session, False, args.retry_failed)
    log(f"stage e8 done: {sum(s['ok'] for s in summaries)} ok, {sum(s['failed'] for s in summaries)} failed, "
        f"{sum(s['skipped'] for s in summaries)} skipped in this session")
    return 0, (time.perf_counter() - t0) / 3600


# ---------------------------------------------------------------- store stage: the whole system at 256K
def store_stage(args, cfg: dict, tcfg: dict) -> int:
    """For each document: cold index build, every page into the int4 store on the GPU, then the question answered from
    the store (KV mode) and from the token ids (text mode), with the reader and the embedder loaded throughout."""
    import numpy as np
    import torch

    from pcm.chunking import page_bounds
    from pcm.env import collect_env
    from pcm.expand import expand_kv, expand_text, kv_layout
    from pcm.health import EXIT_GPU_LOST, GpuTelemetry, classify_failure, interruption_record, preflight
    from pcm.prefill import _sync, generate_greedy
    from pcm.store.kv import encode_page

    env = collect_env()
    if env["git_dirty"]:
        log("uncommitted code: commit first")
        return 5
    exp_dir = REPO_ROOT / "results" / EXP
    out = JsonlLog(exp_dir / "store_256k.jsonl")
    done = out.done_ids(where=lambda r: r.get("status") == "ok")
    sc = cfg["store"]
    L, idx = sc["length"], int(sc["index"])
    todo = []
    for task in sc["tasks"]:
        ex = cell_examples({"task": task, "length": L}, idx + 1, cfg, tcfg)[idx]
        if f"{task}|{L}|{ex['id']}" not in done:
            todo.append((task, ex))
    if not todo:
        log("store: nothing left")
        return 0
    mkv, mt = cfg["_methods"]["P-KV"], cfg["_methods"]["P-T"]
    hkv, ht = config_hash(mkv), config_hash(mt)
    # Answers of the same examples in E8 (P-KV, P-T) and in E6 (P-T reused from A5): the store must not change them.
    e8 = {(r["method_hash"], r["example_id"]): r for r in read_rows(exp_dir) if r["status"] == "ok"}
    e6_text = {r["example_id"]: r for r in read_rows(REPO_ROOT / "results" / "E6_kv") if r["status"] == "ok" and r["method"] == "pcm-text" and r["length"] == L}
    method = R.make_method(mkv)
    method.setup()
    if method.codec is None:
        raise ValueError("the store stage measures the configured codec; pcm_kv.yaml has none")
    pf = preflight(min_free_vram_gib=float(mkv.get("min_free_vram_gib", 10.0)), min_free_ram_gib=float(mkv.get("min_free_ram_gib", 2.0)),
                   min_free_disk_gib=1.0, disk_path=exp_dir)
    log(f"store: {len(todo)} documents; free VRAM {pf['free_vram_gib']:.1f} GiB, RAM {pf['free_ram_gib']:.1f} GiB")
    dev = method.model.device
    block = int(mkv.get("block", 2048))
    with GpuTelemetry(REPO_ROOT / "logs" / EXP / "gpu_telemetry_store.csv"):
        for task, ex in todo:
            rid = f"{task}|{L}|{ex['id']}"
            row = {"id": rid, "task": task, "length": L, "example_id": ex["id"], "git_commit": env["git_commit"], "git_dirty": env["git_dirty"],
                   "method_kv": mkv["name"], "method_kv_hash": hkv, "method_text": mt["name"], "method_text_hash": ht, "codec": method.codec_name}
            store: dict = {}
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                row["resident_before_gib"] = torch.cuda.memory_allocated(dev) / GIB   # reader and embedder weights
                t = time.perf_counter()
                prepared = method.prepare(ex, R.prompt_for(ex, tcfg))
                row["prepare_s"] = time.perf_counter() - t
                units, prefix, suffix = prepared["units"], prepared["prefix"], prepared["suffix"]
                # (1) index: every unit embedded now, cold (the cached copy, which the E8 rows used, is kept for selection)
                _sync(dev)
                t = time.perf_counter()
                emb = method.embedder.encode_documents(prepared["texts"])
                _sync(dev)
                row["index_s"] = time.perf_counter() - t
                if prepared["emb"] is not None:
                    row["index_max_abs_diff_vs_cache"] = float(np.max(np.abs(emb.astype(np.float16).astype(np.float32) - prepared["emb"])))
                else:
                    prepared["emb"] = emb.astype(np.float16).astype(np.float32)
                del emb
                # (2) KV store: every page encoded after P on its own, compacted, kept on the GPU
                P, G = method._shared_caches(prefix, {})
                pb = page_bounds(units.bounds, method.per_page)
                ctx = torch.from_numpy(units.ids.astype(np.int64))
                _sync(dev)
                t = time.perf_counter()
                for p in range(len(pb) - 1):
                    a, b = int(pb[p]), int(pb[p + 1])
                    page = encode_page(method.model, P, ctx[a:b], block=block)
                    store[a] = method.codec.encode(page, len(prefix))
                    del page
                _sync(dev)
                row["kv_store_s"] = time.perf_counter() - t
                row["store_bytes"] = int(sum(e.nbytes for e in store.values())) + int(method.codec.shared_nbytes)
                row["store_allocated_gib"] = torch.cuda.memory_allocated(dev) / GIB
                row["peak_after_ingest_gib"] = torch.cuda.max_memory_allocated(dev) / GIB
                row.update({"doc_tokens": int(units.n_tokens), "units": int(units.n_units), "pages": len(pb) - 1,
                            "text_store_bytes": int(units.n_tokens) * 4, "index_bytes": int(units.n_units) * int(prepared["emb"].shape[1]) * 2 + int(units.n_units + 1) * 4})
                # (3) the question in KV mode, from the store
                max_new = method.max_new(ex)
                _sync(dev)
                t = time.perf_counter()
                s = method.select(prepared)
                pages = sorted(int(x) for x in s["kept_pages"])
                layout = kv_layout([(int(pb[p]), int(pb[p + 1])) for p in pages])
                segs = [method.codec.decode(store[a]) if kind == "page" else G for kind, a, _ in layout]
                cache, n_ctx = expand_kv(P, segs, method.inv_freq, method.temperature)
                bias = (len(prefix), len(prefix) + n_ctx, math.log(method.scale)) if method.scale != 1.0 else None
                gen: dict = {}
                ids = generate_greedy(method.model, cache, suffix, max_new, method.stops, block, timings=gen, context_bias=bias)
                _sync(dev)
                row["kv_question_s"] = time.perf_counter() - t
                pred_kv = method.tokenizer.decode(ids, skip_special_tokens=True).strip()
                del cache, segs
                # (4) the question in text mode, from the token ids
                _sync(dev)
                t = time.perf_counter()
                s2 = method.select(prepared)
                body, _ = expand_text(ctx, s2["spans"], method.gap_ids)
                pred_text, _ = method._generate(ex, torch.cat([prefix, body, suffix]))
                _sync(dev)
                row["text_question_s"] = time.perf_counter() - t
                row.update({"status": "ok", "kept_pages": len(pages), "context_tokens": int(n_ctx), "prediction_kv": pred_kv, "prediction_text": pred_text,
                            "score_kv": R.metrics.score(pred_kv, ex), "score_text": R.metrics.score(pred_text, ex),
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / GIB, "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / GIB,
                            "vram_gib": float(sc["vram_gib"])})
                row["inside_card"] = row["peak_reserved_gib"] <= row["vram_gib"]
                rk, rt = e8.get((hkv, ex["id"])), e8.get((ht, ex["id"])) or e6_text.get(ex["id"])
                row["same_as_e8_kv"] = None if rk is None else rk["prediction"] == pred_kv
                row["same_as_text_row"] = None if rt is None else rt["prediction"] == pred_text
            except Exception as e:  # noqa: BLE001 - classified like the runner's failures
                kind = classify_failure(e)
                if kind == "gpu_lost":
                    atomic_write_json(exp_dir / "interrupted_store.json", {**interruption_record(e, None), "task": task, "length": L})
                    log(f"GPU lost during the store of {ex['id']}: {e}. Rerun the same command once the GPU is back.")
                    raise SystemExit(EXIT_GPU_LOST)
                row.update({"status": "failed", "failure_kind": kind, "error": f"{type(e).__name__}: {str(e)[:500]}",
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(dev) / GIB, "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / GIB})
            finally:
                store.clear()
                gc.collect()
                torch.cuda.empty_cache()
            row["timestamp_utc"] = R._now()
            out.append(row)
            if row["status"] == "ok":
                log(f"store {ex['id']}: index {row['index_s']:.1f} s, KV store {row['kv_store_s']:.1f} s, {row['store_bytes'] / GIB:.2f} GiB; "
                    f"question KV {row['kv_question_s']:.2f} s, text {row['text_question_s']:.2f} s; peak {row['peak_allocated_gib']:.1f} GiB "
                    f"allocated, {row['peak_reserved_gib']:.1f} GiB reserved; same as E8 KV row: {row['same_as_e8_kv']}, as text row: {row['same_as_text_row']}")
            else:
                log(f"store {ex['id']}: {row['status']} ({row.get('failure_kind')}): {row.get('error')}")
    return 0


def main() -> int:
    global _LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None, help="labels (P-T, P-KV, M2) or method names")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    cfg, tcfg = load_config(), load_task_config()
    left = args.max_hours
    for st in (STAGES if args.stage == "all" else (args.stage,)):
        _LOG_FILE = REPO_ROOT / "logs" / "A8" / f"{st}.log"
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if st == "e8":
            status, spent = e8_stage(args, cfg, tcfg, left)
            if left is not None:
                left -= spent
        elif args.pilot:
            continue
        else:
            status = store_stage(args, cfg, tcfg)
        if status:
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
