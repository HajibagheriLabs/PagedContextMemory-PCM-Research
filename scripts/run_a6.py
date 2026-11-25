"""Run A6 (configs/a6_kv.yaml): KV expansion against text expansion on the same pages. Resumable.

    python scripts/run_a6.py --stage tune --max-hours 2        # APE grid on the synthetic train split
    python scripts/run_a6.py --stage e6 --pilot                # E6 cells: pilot rows and projection
    python scripts/run_a6.py --stage e6 --max-hours 12         # E6: (a) text, (b) KV, (c) KV + fix, (d) page sizes
    python scripts/run_a6.py --stage deviation                 # KV and attention deviation, 20 examples
    python scripts/run_a6.py --stage store                     # whole-document page store at 256K
    python scripts/run_a6.py --stage all --max-hours 16        # the four in order

The page selection is configs/pcm_text.yaml (A5). The tuning stage needs the synthetic train split
(scripts/make_data.py --synth --split train ...) and writes results/E6_kv/ape_choice.json once, by the rule of
the config; E6's "kv-ape" mode uses it. Tuning rows go to results/E6_kv_tune, never beside test rows.
A lost GPU stops the run with exit status 75; rerun the same command once the GPU is back.
Output also goes to logs/A6/<stage>.log.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcm.eval import runner as R  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.common import load_task_config  # noqa: E402
from run_a4 import Session, cell_examples, cell_key, cell_stats, read_rows  # noqa: E402
from run_a5 import expand_tasks, lengths_for, project  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "a6_kv.yaml"
STAGES = ("tune", "e6", "deviation", "store")
EXP, TUNE_EXP = "E6_kv", "E6_kv_tune"
_LOG_FILE = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d ") + line + "\n")


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    p = REPO_ROOT / cfg["pcm_text"]
    if not p.is_file():
        raise FileNotFoundError(f"{p} is missing: it is fixed at the end of A5")
    cfg["_pcm_text"] = yaml.safe_load(p.read_text(encoding="utf-8"))["pcm_text"]
    return cfg


# ---------------------------------------------------------------- method configurations
def mode_config(cfg: dict, mode: str, page: int | None = None, ape: dict | None = None, suffix: str = "") -> dict:
    """text: configs/pcm_text.yaml as it is (same configuration, same hash, as in A5); kv: the same pages in KV
    mode; kv-ape: KV mode with the tuned temperature and scale. `page` overrides the page size (d)."""
    base = dict(cfg["_pcm_text"])
    name = "pcm-text"
    if page is not None and int(page) != int(base["page"]):
        base["page"] = int(page)
        name += f"-p{page}"
    if mode == "text":
        m = {**base, "name": name + suffix}
    elif mode in ("kv", "kv-ape"):
        t, s = (1.0, 1.0) if mode == "kv" else (float(ape["temperature"]), float(ape["scale"]))
        m = {**base, "method": "pcm_kv", "kv": {"temperature": t, "scale": s, "store_gpu_gib": float(cfg["kv"]["store_gpu_gib"])},
             "name": name.replace("pcm-text", "pcm-kv" if mode == "kv" else "pcm-kv-ape") + suffix}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return m


def ape_choice(cfg: dict) -> dict | None:
    p = REPO_ROOT / "results" / EXP / "ape_choice.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def tune_cells(cfg: dict) -> list[dict]:
    tr = cfg["ape"]["train"]
    out = []
    for t in cfg["ape"]["temperatures"]:
        for s in cfg["ape"]["scales"]:
            m = mode_config(cfg, "kv-ape", ape={"temperature": t, "scale": s}, suffix=f"-T{t:g}-S{s:g}-train")
            for task in tr["tasks"]:
                out.append({"method": m["name"], "mcfg": m, "task": task, "length": tr["length"], "n": int(tr["n"]), "split": "train",
                            "seed": int(tr["seed"]), "T": float(t), "S": float(s)})
    return out


def e6_cells(cfg: dict, ape: dict | None) -> list[dict]:
    out = []

    def add(m, task, L):
        out.append({"method": m["name"], "mcfg": m, "task": task, "length": L, "n": int(cfg["n"])})

    main = cfg["e6"]["main"]
    for mode in main["modes"]:
        if mode == "kv-ape" and ape is None:
            continue
        m = mode_config(cfg, mode, ape=ape)
        seen = set()
        for L in main["lengths"]:
            for t in expand_tasks(cfg, main["tasks"]):
                for LL in lengths_for(t, [L]):
                    if (t, LL) not in seen:
                        seen.add((t, LL))
                        add(m, t, LL)
    pg = cfg["e6"]["pages"]
    for size in pg["sizes"]:
        for mode in pg["modes"]:
            if mode == "kv-ape" and ape is None:
                continue
            m = mode_config(cfg, mode, page=size, ape=ape)
            for L in pg["lengths"]:
                for t in expand_tasks(cfg, pg["tasks"]):
                    add(m, t, L)
    seen, uniq = set(), []
    for c in out:
        k = (c["method"], c["task"], c["length"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


def choose_ape(cfg: dict, rows: list[dict]) -> dict:
    """The rule of the config: highest mean train accuracy; ties: smaller |1-T| + |1-S|, larger T, larger S."""
    from pcm.tasks.common import config_hash

    tr = cfg["ape"]["train"]
    by = {}
    for r in rows:
        if r["status"] == "ok" and r.get("score") is not None:
            by.setdefault((r["method_hash"], r["task"]), []).append(r["score"])
    table = []
    for t in cfg["ape"]["temperatures"]:
        for s in cfg["ape"]["scales"]:
            h = config_hash(mode_config(cfg, "kv-ape", ape={"temperature": t, "scale": s}, suffix=f"-T{t:g}-S{s:g}-train"))
            accs = {}
            for task in tr["tasks"]:
                sc = by.get((h, task), [])
                if len(sc) < int(tr["n"]):
                    raise RuntimeError(f"tuning incomplete at T={t}, S={s}, {task}: {len(sc)}/{tr['n']}")
                accs[task] = sum(sc) / len(sc)
            table.append({"temperature": float(t), "scale": float(s), "accuracy": sum(accs.values()) / len(accs), "tasks": accs})
    table.sort(key=lambda x: (-x["accuracy"], abs(1 - x["temperature"]) + abs(1 - x["scale"]), -x["temperature"], -x["scale"]))
    return {"rule": "highest mean train accuracy; ties: smaller |1-T|+|1-S|, larger T, larger S",
            "temperature": table[0]["temperature"], "scale": table[0]["scale"], "ranking": table}


# ---------------------------------------------------------------- runner stages
def examples_for(c: dict, cfg: dict, tcfg: dict) -> list[dict]:
    if c.get("split") == "train":
        return R.load_examples(c["task"], c["length"], c["n"], c["seed"], tcfg, split="train")
    return cell_examples(c, c["n"], cfg, tcfg)


def run_cells(todo, exp, cfg, tcfg, stats, session, pilot, answer_from=None, retry_failed=False):
    summaries = []
    for c in todo:
        s = stats.get(cell_key(c), {"done": 0, "failed": 0})
        done = s["done"] - (s["failed"] if retry_failed else 0)
        if done >= (cfg["pilot_n"] if pilot else c["n"]):
            continue
        examples = examples_for(c, cfg, tcfg)
        inst = session.method_for(c["mcfg"])
        summaries.append(R.run(c["mcfg"], c["task"], c["length"], None, seed=c.get("seed", cfg["seed"]), exp=exp, pilot=pilot,
                               pilot_n=max(0, cfg["pilot_n"] - done) if pilot else cfg["pilot_n"], retry_failed=retry_failed,
                               examples=examples, method=inst, cfg=tcfg, log=log, answer_from=answer_from))
        inst = examples = None
        session.current = None
    return summaries


def runner_stage(stage: str, args, cfg: dict, tcfg: dict, hours_left: float | None) -> tuple[int, float]:
    t0 = time.perf_counter()
    if stage == "tune":
        exp, todo, answer_from = TUNE_EXP, tune_cells(cfg), None
    else:
        ape = ape_choice(cfg)
        if ape is None:
            log("E6 needs results/E6_kv/ape_choice.json: run --stage tune first (the kv-ape cells are left out until then)")
            if not args.pilot:
                return 4, 0.0
        exp, todo, answer_from = EXP, e6_cells(cfg, ape), cfg["e6"].get("answer_from")
    if args.tasks:
        todo = [c for c in todo if c["task"] in args.tasks]
    if args.lengths:
        todo = [c for c in todo if c["length"] in args.lengths]
    if args.methods:
        todo = [c for c in todo if c["method"] in args.methods]
    exp_dir = REPO_ROOT / "results" / exp
    log(f"stage {stage} ({exp}): {len(todo)} cells{' (pilot only)' if args.pilot else ''}")
    session = Session()
    run_cells(todo, exp, cfg, tcfg, cell_stats(read_rows(exp_dir)), session, True, answer_from)
    stats = cell_stats(read_rows(exp_dir))
    proj = project(todo, stats)
    by_method: Counter = Counter()
    for pc in proj["cells"]:
        by_method[pc["method"]] += pc["remaining_s"]
    proj.update({"stage": stage, "by_method_h": {k: round(v / 3600, 2) for k, v in sorted(by_method.items())}})
    filtered = bool(args.tasks or args.lengths or args.methods)
    atomic_write_json(exp_dir / (f"projection_{stage}_partial.json" if filtered else f"projection_{stage}.json"), proj)
    log(f"stage {stage}: projected {proj['remaining_h']:.2f} h for what is left")
    if args.pilot:
        return 0, (time.perf_counter() - t0) / 3600
    limit = hours_left if hours_left is not None else float(cfg["ask_above_hours"])
    if proj["remaining_h"] > limit:
        log(f"stage {stage} is projected at {proj['remaining_h']:.1f} h, above the {limit:.1f} h allowed: rerun with a larger --max-hours")
        return 3, (time.perf_counter() - t0) / 3600
    summaries = run_cells(todo, exp, cfg, tcfg, stats, session, False, answer_from, args.retry_failed)
    log(f"stage {stage} done: {sum(s['ok'] for s in summaries)} ok, {sum(s['failed'] for s in summaries)} failed, "
        f"{sum(s['skipped'] for s in summaries)} skipped in this session")
    if stage == "tune" and not filtered:
        path = REPO_ROOT / "results" / EXP / "ape_choice.json"
        if not path.is_file():
            choice = choose_ape(cfg, read_rows(exp_dir))
            atomic_write_json(path, choice)
            log(f"APE choice: T = {choice['temperature']}, S = {choice['scale']} (train accuracy {choice['ranking'][0]['accuracy']:.3f})")
    return 0, (time.perf_counter() - t0) / 3600


# ---------------------------------------------------------------- deviation and store stages
def _prepared(method, ex, tcfg):
    return method.prepare(ex, R.prompt_for(ex, tcfg))


def _caches(method, prepared):
    """(sequential cache of P + context, stitched cache, prefix length, context length, question ids, layout)."""
    import numpy as np
    import torch

    from pcm.chunking import page_bounds
    from pcm.expand import expand_kv, expand_text, kv_layout
    from pcm.prefill import prefill_in_blocks
    from pcm.store.kv import encode_page

    s = method.select(prepared)
    units = prepared["units"]
    ctx = torch.from_numpy(units.ids.astype(np.int64))
    body, _ = expand_text(ctx, s["spans"], method.gap_ids)
    prefix = prepared["prefix"]
    seq = prefill_in_blocks(method.model, torch.cat([prefix, body]))
    P, G = method._shared_caches(prefix, {})
    pb = page_bounds(units.bounds, method.per_page)
    layout = kv_layout([(int(pb[p]), int(pb[p + 1])) for p in sorted(int(x) for x in s["kept_pages"])])
    segs = [encode_page(method.model, P, ctx[a:b]) if kind == "page" else G for kind, a, b in layout]
    st, n_ctx = expand_kv(P, segs, method.inv_freq)
    return seq, st, len(prefix), n_ctx, prepared["suffix"], layout


def deviation_stage(args, cfg: dict, tcfg: dict) -> int:
    import torch

    from pcm.env import collect_env
    from pcm.eval import kv_fidelity as F

    env = collect_env()
    if env["git_dirty"]:
        log("uncommitted code: commit first")
        return 5
    exp_dir = REPO_ROOT / "results" / EXP
    out = JsonlLog(exp_dir / "deviation.jsonl")
    done = out.done_ids()
    ape = ape_choice(cfg) or {"temperature": 1.0, "scale": 1.0}
    method = R.make_method(mode_config(cfg, "kv"))
    method.setup()
    dv = cfg["deviation"]
    for task in dv["tasks"]:
        for ex in cell_examples({"task": task, "length": dv["length"]}, int(dv["per_task"]), cfg, tcfg):
            if ex["id"] in done:
                continue
            t0 = time.perf_counter()
            prepared = _prepared(method, ex, tcfg)
            seq, st, p, n_ctx, q, layout = _caches(method, prepared)
            att_seq, lg_seq = F.question_pass(method.model, seq, q)
            att_st, lg_st = F.question_pass(method.model, st, q)
            row = {"id": ex["id"], "task": task, "length": dv["length"], "git_commit": env["git_commit"], "git_dirty": env["git_dirty"],
                   "prefix_tokens": p, "context_tokens": n_ctx, "question_tokens": int(len(q)),
                   "pages": sum(1 for k, _, _ in layout if k == "page"), "segments": len(layout),
                   "kv_deviation": F.kv_deviation(seq, st, p, p + n_ctx),
                   "attention_deviation": F.attention_deviation(att_seq, att_st), "next_token": F.next_token_agreement(lg_seq, lg_st)}
            del att_st
            if (ape["temperature"], ape["scale"]) != (1.0, 1.0):
                fixed = st.scale_keys(p, p + n_ctx, 1.0 / ape["temperature"])
                bias = (p, p + n_ctx, math.log(ape["scale"])) if ape["scale"] != 1.0 else None
                att_fix, lg_fix = F.question_pass(method.model, fixed, q, bias)
                row["ape"] = {"temperature": ape["temperature"], "scale": ape["scale"],
                              "attention_deviation": F.attention_deviation(att_seq, att_fix), "next_token": F.next_token_agreement(lg_seq, lg_fix)}
                del att_fix, fixed
            row["seconds"] = time.perf_counter() - t0
            out.append(row)
            del seq, st, att_seq
            torch.cuda.empty_cache()
            log(f"deviation {ex['id']}: layer-27 K {row['kv_deviation'][-1]['k']:.3f}, attention TV last layer "
                f"{row['attention_deviation'][-1]:.3f}, KL {row['next_token']['kl']:.4f}")
    return 0


def store_stage(args, cfg: dict, tcfg: dict) -> int:
    """Every page of a 256K document encoded into the page store (GPU up to the limit, then disk), then the
    example's selected pages read back and stitched, against recomputing the same context from text."""
    import numpy as np
    import torch

    from pcm.chunking import page_bounds
    from pcm.env import collect_env
    from pcm.expand import expand_kv, expand_text, kv_layout
    from pcm.prefill import _sync, prefill_in_blocks
    from pcm.store.kv import encode_page
    from pcm.store.pages import PageStore

    env = collect_env()
    if env["git_dirty"]:
        log("uncommitted code: commit first")
        return 5
    out = JsonlLog(REPO_ROOT / "results" / EXP / "store_bench.jsonl")
    done = out.done_ids()
    method = R.make_method(mode_config(cfg, "kv"))
    method.setup()
    sc = cfg["store"]
    dev = method.model.device
    for task in sc["tasks"]:
        for ex in cell_examples({"task": task, "length": sc["length"]}, int(sc["per_task"]), cfg, tcfg):
            if ex["id"] in done:
                continue
            prepared = _prepared(method, ex, tcfg)
            units = prepared["units"]
            ctx = torch.from_numpy(units.ids.astype(np.int64))
            P, G = method._shared_caches(prepared["prefix"], {})
            pb = page_bounds(units.bounds, method.per_page)
            store = PageStore(method.store_gpu_bytes, disk_dir=REPO_ROOT / sc["disk_dir"] / ex["id"], device=dev)
            torch.cuda.reset_peak_memory_stats()
            try:
                _sync(dev)
                t0 = time.perf_counter()
                for p in range(len(pb) - 1):
                    store.put(int(pb[p]), encode_page(method.model, P, ctx[int(pb[p]):int(pb[p + 1])]))
                _sync(dev)
                ingest_s = time.perf_counter() - t0
                s = method.select(prepared)
                layout = kv_layout([(int(pb[p]), int(pb[p + 1])) for p in sorted(int(x) for x in s["kept_pages"])])
                on_disk = sum(1 for k, a, _ in layout if k == "page" and store.where(a) == "disk")
                _sync(dev)
                t1 = time.perf_counter()
                segs = [store.get(a) if k == "page" else G for k, a, _ in layout]
                cache, n_ctx = expand_kv(P, segs, method.inv_freq)
                _sync(dev)
                expand_s = time.perf_counter() - t1
                del cache, segs
                body, _ = expand_text(ctx, s["spans"], method.gap_ids)
                _sync(dev)
                t2 = time.perf_counter()
                rec = prefill_in_blocks(method.model, torch.cat([prepared["prefix"], body]))
                _sync(dev)
                recompute_s = time.perf_counter() - t2
                del rec
                row = {"id": ex["id"], "task": task, "length": sc["length"], "git_commit": env["git_commit"], "git_dirty": env["git_dirty"],
                       "doc_tokens": units.n_tokens, "pages": len(pb) - 1, "page": method.page,
                       "ingest_s": ingest_s, "ingest_tokens_per_s": units.n_tokens / ingest_s,
                       "store_gpu_bytes": store.bytes_gpu, "store_disk_bytes": store.bytes_disk, "disk_write_s": store.write_s,
                       "bytes_per_token": (store.bytes_gpu + store.bytes_disk) / units.n_tokens,
                       "selected_pages": sum(1 for k, _, _ in layout if k == "page"), "selected_on_disk": on_disk,
                       "context_tokens": int(n_ctx), "expand_s": expand_s, "recompute_s": recompute_s,
                       "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3}
                out.append(row)
                log(f"store {ex['id']}: ingest {ingest_s:.1f} s, {row['store_gpu_bytes'] / 1024**3:.1f} GiB GPU + "
                    f"{row['store_disk_bytes'] / 1024**3:.1f} GiB disk; expand {expand_s * 1000:.0f} ms vs recompute {recompute_s * 1000:.0f} ms")
            finally:
                store.clear()
                torch.cuda.empty_cache()
    return 0


def main() -> int:
    global _LOG_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=STAGES + ("all",))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    cfg, tcfg = load_config(), load_task_config()
    left = args.max_hours
    for st in (STAGES if args.stage == "all" else (args.stage,)):
        _LOG_FILE = REPO_ROOT / "logs" / "A6" / f"{st}.log"
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if st in ("tune", "e6"):
            status, spent = runner_stage(st, args, cfg, tcfg, left)
            if left is not None:
                left -= spent
        elif args.pilot:
            continue
        else:
            status = deviation_stage(args, cfg, tcfg) if st == "deviation" else store_stage(args, cfg, tcfg)
        if status:
            return status
    return 0


if __name__ == "__main__":
    sys.exit(main())
