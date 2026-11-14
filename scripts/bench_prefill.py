"""Prefill throughput: tokens per second of chunked prefill at several lengths, per model.

Settings come from configs/a1_engine.yaml (section `throughput`). Each timed run
appends one line to results/<experiment>/throughput.jsonl; a rerun skips runs
already present. Also writes results/<experiment>/env.json.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

from pcm.env import collect_env  # noqa: E402
from pcm.hub import load_assets  # noqa: E402
from pcm.models import load_reader  # noqa: E402
from pcm.prefill import prefill_in_blocks  # noqa: E402
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "a1_engine.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tp, block = cfg["throughput"], cfg["block"]
    dtype = getattr(torch, tp["dtype"])
    out_dir = REPO_ROOT / "results" / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "throughput.jsonl"
    (out_dir / "env.json").write_text(json.dumps(collect_env(), indent=2), encoding="utf-8")

    done = set()
    if runs_path.is_file():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    assets = load_assets()["models"]
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    print(f"free VRAM {free_vram / GIB:.1f} of {total_vram / GIB:.1f} GiB")
    for key in tp["models"]:
        repo = assets.get(key, key)
        todo = [
            (n, r)
            for n in tp["lengths"]
            for r in range(tp["repeats"])
            if f"{repo}|{n}|{r}" not in done
        ]
        if not todo:
            print(f"{repo}: all runs present, skipped")
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
                **res,
            }
            with runs_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"{repo:18s} {n:6d} tokens  run {r}  {res['tokens_per_second']:8.0f} tok/s  "
                f"{res['seconds']:6.1f} s  peak {res['peak_vram_gib']:.2f} GiB "
                f"(reserved {res['peak_reserved_gib']:.2f})"
            )
        del model
        free()
    print(f"\nwrote {runs_path}")


if __name__ == "__main__":
    main()
