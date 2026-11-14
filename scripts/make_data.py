"""Materialise the task data of section 5.2 and measure the benchmarks. Resumable: rerun to continue.

    python scripts/make_data.py --synth                 # S1..S4, every grid length, seed 0, 100 per cell
    python scripts/make_data.py --synth --lengths 8k 16k --seeds 0 1 2
    python scripts/make_data.py --synth --split train --seeds 1000 --n 500
    python scripts/make_data.py --benchmarks            # Qwen token lengths (and located facts) of BABILong and InfiniteBench
    python scripts/make_data.py --sizes                 # bytes on disk of every dataset
    python scripts/make_data.py --verify                # regenerate a sample of every stored synthetic file and compare

Data go to data/ (git-ignored). Small summaries that reports quote go to
results/A2_tasks/: synth_manifest.jsonl (one line per synthetic example, no text),
lengths_babilong.jsonl, lengths_infbench.jsonl, sizes.json, verify_synth.json,
make_data_env_<mode>.json (latest run) and make_data_env_history.jsonl (every run).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.env import collect_env  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.common import get_tokenizer, iter_jsonl, load_task_config, repo_path, tokenizer_tag  # noqa: E402

OUT = REPO_ROOT / "results" / "A2_tasks"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def synth(cfg: dict, tasks: list[str], lengths: list[str], seeds: list[int], n: int, split: str) -> None:
    from pcm.tasks.synth import BookPool, data_file, ensure_book_cache, example_id, generate_example

    tok = get_tokenizer(cfg["tokenizer"])
    pool = BookPool(ensure_book_cache(tok, cfg, log=log))
    manifest = JsonlLog(OUT / ("synth_manifest.jsonl" if split == "test" else "synth_manifest_train.jsonl"))
    in_manifest = manifest.done_ids()
    for seed in seeds:
        for length in lengths:
            target = cfg["lengths"].get(length) or cfg["phase_b_lengths"][length]
            for task in tasks:
                path = data_file(task, length, seed, cfg, split=split)
                data = JsonlLog(path)
                have = data.done_ids()
                t0 = time.perf_counter()
                made = 0
                for i in range(n):
                    ex = None
                    ex_id = example_id(split, task, length, seed, i)
                    if ex_id not in have:
                        ex = generate_example(task, target, seed, i, tok, pool, split=split, length_name=length, cfg=cfg)
                        data.append(ex)
                        made += 1
                    if ex_id not in in_manifest:
                        if ex is None:  # data written before the manifest line: read it back
                            ex = next(r for r in iter_jsonl(path) if r["id"] == ex_id)
                        manifest.append({
                            "id": ex["id"], "task": task, "family": ex["family"], "split": split, "seed": seed,
                            "length_name": length, "length_nominal": target, "length_tokens": ex["length_tokens"],
                            "depth_decile": ex["depth_decile"], "attempts": ex["attempts"],
                            "evidence": [{k: e[k] for k in ("hop", "tok_start", "tok_end", "decile")} for e in ex["evidence"]],
                            "n_distractors": len(ex["distractors"]), "books": [p["book"] for p in ex["haystack"]],
                            "answers": ex["answers"],
                        })
                        in_manifest.add(ex_id)
                dt = time.perf_counter() - t0
                log(f"synth {split} {task} {length} seed {seed}: {made} new examples in {dt:.1f} s -> {path.relative_to(REPO_ROOT)}")


def benchmarks(cfg: dict) -> None:
    from pcm.tasks import babilong, infbench

    tok = get_tokenizer(cfg["tokenizer"])
    rows = []
    for task in cfg["babilong"]["tasks"]:
        for length in cfg["babilong"]["lengths"]:
            t0 = time.perf_counter()
            meta = babilong.ensure_meta(task, length, tok, cfg)
            for i in sorted(meta):
                m = meta[i]
                rows.append({"id": f"babilong-{task}-{length}-{i:03d}", "task": task, "length_name": length, "index": i,
                             "length_tokens": m["length_tokens"], "prompt_tokens": m["prompt_tokens"], "chars": m["chars"],
                             "n_facts": len(m["facts"]), "locator": m.get("locator")})
            log(f"babilong {task} {length}: {len(meta)} rows ({time.perf_counter() - t0:.1f} s)")
    _write_jsonl(OUT / "lengths_babilong.jsonl", rows, tok)

    rows = []
    for task in cfg["infbench"]["tasks"]:
        t0 = time.perf_counter()
        meta = infbench.ensure_meta(task, tok, cfg, log=log)
        for rid, m in meta.items():
            rows.append({"id": f"infbench-{task}-{rid}", "task": task, "length_tokens": m["length_tokens"],
                         "prompt_tokens": m["prompt_tokens"], "chars": m["chars"], "n_evidence": len(m["evidence"]),
                         "evidence_decile": m["evidence"][0]["decile"] if m["evidence"] else None})
        log(f"infbench {task}: {len(meta)} rows ({time.perf_counter() - t0:.1f} s)")
    _write_jsonl(OUT / "lengths_infbench.jsonl", rows, tok)


def _write_jsonl(path: Path, rows: list[dict], tok) -> None:
    import json

    from pcm.runlog import atomic_write_text

    text = "".join(json.dumps({**r, "tokenizer": tokenizer_tag(tok)}, ensure_ascii=False) + "\n" for r in rows)
    atomic_write_text(path, text)


def verify(cfg: dict, indices: list[int]) -> None:
    """Regenerate examples `indices` of every stored synthetic file with the current code; they must be identical."""
    import json

    from pcm.env import collect_env
    from pcm.tasks.synth import BookPool, ensure_book_cache, generate_example

    tok = get_tokenizer(cfg["tokenizer"])
    pool = BookPool(ensure_book_cache(tok, cfg, log=log))
    root = repo_path(cfg["synth"]["data_dir"])
    checked, mismatched = 0, []
    for path in sorted(root.glob("*/*_s*.jsonl")):
        task, (length, seed) = path.parent.name, path.stem.rsplit("_s", 1)
        target = cfg["lengths"].get(length) or cfg["phase_b_lengths"][length]
        stored = {i: r for i, r in enumerate(iter_jsonl(path)) if i in indices}
        for i, row in stored.items():
            again = generate_example(task, target, int(seed), i, tok, pool, length_name=length, cfg=cfg)
            checked += 1
            if json.dumps(again, sort_keys=True) != json.dumps(row, sort_keys=True):
                mismatched.append(row["id"])
        log(f"verify {path.relative_to(REPO_ROOT)}: {len(stored)} regenerated")
    env = collect_env()
    atomic_write_json(OUT / "verify_synth.json", {"code": {"git_commit": env["git_commit"], "git_dirty": env["git_dirty"]},
                                                  "indices": indices, "checked": checked, "identical": checked - len(mismatched), "mismatched": mismatched})
    log(f"verify: {checked - len(mismatched)} of {checked} regenerated examples identical to the stored ones")


def _tree_bytes(p: Path) -> tuple[int, int]:
    if p.is_file():
        return 1, p.stat().st_size
    files = [f for f in p.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def sizes(cfg: dict) -> None:
    from pcm.hub import dataset_path, load_assets
    from pcm.tasks.synth import SYNTH_TASKS

    a = load_assets()["datasets"]
    out: dict = {"synth": {}, "babilong": {}, "infbench": {}, "pg19": {}, "caches": {}}
    root = repo_path(cfg["synth"]["data_dir"])
    for task in SYNTH_TASKS:
        for f in sorted((root / task).glob("*.jsonl")) if (root / task).is_dir() else []:
            out["synth"][f"{task}/{f.name}"] = f.stat().st_size
    b = dataset_path(a["babilong"]["repo"], [f"data/{t}/{L}.json" for t in cfg["babilong"]["tasks"] for L in cfg["babilong"]["lengths"]])
    for L in cfg["babilong"]["lengths"]:
        out["babilong"][L] = sum((b / "data" / t / f"{L}.json").stat().st_size for t in cfg["babilong"]["tasks"])
    ib = dataset_path(a["infbench"]["repo"], a["infbench"]["files"])
    for f in a["infbench"]["files"]:
        out["infbench"][f] = (ib / f).stat().st_size
    pg = dataset_path(a["pg19"]["repo"], [f"data/{s}-*.parquet" for s in a["pg19"]["splits"]])
    for s in a["pg19"]["splits"]:
        out["pg19"][s] = sum(f.stat().st_size for f in (pg / "data").glob(f"{s}-*.parquet"))
    for name, rel in (("pg19_token_cache", cfg["synth"]["cache_dir"]), ("babilong_meta", cfg["babilong"]["meta_dir"]), ("infbench_meta", cfg["infbench"]["meta_dir"])):
        p = repo_path(rel)
        out["caches"][name] = _tree_bytes(p)[1] if p.exists() else 0
    atomic_write_json(OUT / "sizes.json", out)
    log(f"sizes written to {(OUT / 'sizes.json').relative_to(REPO_ROOT)}")


def main() -> int:
    from pcm.tasks.synth import SYNTH_TASKS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--benchmarks", action="store_true")
    ap.add_argument("--sizes", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--verify-indices", nargs="*", type=int, default=[0, 9, 50, 99])
    ap.add_argument("--tasks", nargs="*", default=list(SYNTH_TASKS))
    ap.add_argument("--lengths", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default="test", choices=["test", "train"])
    args = ap.parse_args()
    cfg = load_task_config()
    env = collect_env()
    modes = [m for m in ("synth", "benchmarks", "sizes", "verify") if getattr(args, m)]
    # One file per mode, so two modes can run at the same time in separate processes; plus a history of every run.
    record = {**env, "argv": sys.argv[1:]}
    atomic_write_json(OUT / f"make_data_env_{'_'.join(modes) or 'none'}.json", record)
    JsonlLog(OUT / "make_data_env_history.jsonl", id_field="timestamp_utc").append(record)
    if env["git_dirty"]:
        log("warning: uncommitted code; the data will not be traceable to a commit")
    if args.synth:
        synth(cfg, args.tasks, args.lengths or list(cfg["lengths"]), args.seeds or cfg["synth"]["materialise_seeds"],
              args.n or cfg["synth"]["n_per_cell"], args.split)
    if args.benchmarks:
        benchmarks(cfg)
    if args.sizes:
        sizes(cfg)
    if args.verify:
        verify(cfg, args.verify_indices)
    return 0


if __name__ == "__main__":
    sys.exit(main())
