"""Prefill throughput: tokens per second of chunked prefill at several lengths, per model.

Settings come from configs/a1_engine.yaml (section `throughput`). Each timed run
appends one durable line to results/<experiment>/throughput.jsonl, stamped with
the code commit. A rerun skips the runs already measured on the current commit,
so after a crash the same command continues where it stopped. If the GPU is
lost, the script writes results/<experiment>/interrupted_throughput.json and
exits with status 75. GPU temperature and power are sampled to
logs/<experiment>/gpu_telemetry.csv while it runs.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from pcm.env import collect_env  # noqa: E402
from pcm.health import (  # noqa: E402
    EXIT_GPU_LOST,
    GpuLost,
    GpuTelemetry,
    classify_failure,
    interruption_record,
    preflight,
)
from pcm.hub import load_assets  # noqa: E402
from pcm.models import load_reader  # noqa: E402
from pcm.prefill import prefill_in_blocks  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.pg19 import first_long_book  # noqa: E402

GIB = 1024**3


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def timed_prefill(model, ids, block: int) -> dict:
    free()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cache = prefill_in_blocks(model, ids, block=block)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    out = {
        "seconds": seconds,
        "tokens_per_second": len(ids) / seconds,
        "cache_gib": cache.nbytes / GIB,
        "allocated_before_gib": before / GIB,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB,
    }
    del cache
    free()
    return out


def run(cfg: dict, out_dir: Path, log: JsonlLog, env: dict, state: dict) -> None:
    tp, block = cfg["throughput"], cfg["block"]
    dtype = getattr(torch, tp["dtype"])
    commit = env["git_commit"]
    # Only runs measured on this exact, clean code count as done.
    done = set() if env["git_dirty"] else log.done_ids(
        where=lambda r: r.get("git_commit") == commit and r.get("git_dirty") is False
    )
    assets = load_assets()["models"]
    for key in tp["models"]:
        repo = assets.get(key, key)
        todo = [(n, r) for n in tp["lengths"] for r in range(tp["repeats"]) if f"{repo}|{n}|{r}" not in done]
        if not todo:
            print(f"{repo}: all runs present for commit {str(commit)[:10]}, skipped")
            continue
        model, tokenizer = load_reader(repo, dtype=dtype)
        book = first_long_book(tokenizer, cfg["text"]["split"], cfg["text"]["min_tokens"])
        ids = torch.tensor(book["ids"], dtype=torch.long)
        timed_prefill(model, ids[: tp["warmup_tokens"]], block)  # warm-up, not recorded
        for n, r in todo:
            res = timed_prefill(model, ids[:n], block)
            row = {
                "id": f"{repo}|{n}|{r}",
                "model": repo,
                "dtype": tp["dtype"],
                "tokens": n,
                "block": block,
                "repeat": r,
                "book": book["title"],
                "git_commit": commit,
                "git_dirty": env["git_dirty"],
                **res,
            }
            log.append(row)
            state["last_done"] = row["id"]
            print(
                f"{repo:18s} {n:6d} tokens  run {r}  {res['tokens_per_second']:8.0f} tok/s  "
                f"{res['seconds']:6.1f} s  peak {res['peak_vram_gib']:.2f} GiB "
                f"(reserved {res['peak_reserved_gib']:.2f})"
            )
        del model
        free()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "a1_engine.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "results" / cfg["experiment"]
    log = JsonlLog(out_dir / "throughput.jsonl")
    marker = out_dir / "interrupted_throughput.json"
    state = {"last_done": None}
    try:
        report = preflight(min_free_vram_gib=cfg["throughput"]["min_free_vram_gib"], min_free_disk_gib=1.0)
        print(f"free VRAM {report['free_vram_gib']:.1f} GiB, free RAM {report['free_ram_gib']:.1f} GiB")
        env = collect_env()
        atomic_write_json(out_dir / "env.json", env)
        with GpuTelemetry(REPO_ROOT / "logs" / cfg["experiment"] / "gpu_telemetry.csv"):
            run(cfg, out_dir, log, env, state)
    except Exception as exc:  # noqa: BLE001 - classified below, re-raised unless the GPU was lost
        if classify_failure(exc) != "gpu_lost" and not isinstance(exc, GpuLost):
            raise
        atomic_write_json(marker, interruption_record(exc, last_done=state["last_done"]))
        print(f"GPU LOST: {exc}\nCompleted runs are saved. Rerun this command once the GPU is back.")
        sys.exit(EXIT_GPU_LOST)
    if marker.is_file():
        marker.unlink()  # a completed run clears the marker of an earlier interruption
    print(f"\nwrote {log.path}")


if __name__ == "__main__":
    main()
