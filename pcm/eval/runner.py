"""Evaluation runner: one method on one task cell, one durable JSON line per example.

    run(method_config, task, length, n, seed, exp=...)

task:   "synth:<S1|S2_multikey|...>", "babilong:<qa1..qa10>" or "infbench:<passkey|...>"
length: a grid name ("8k" ... "256k"; BABILong also "0k".."4k"); "full" for InfiniteBench
method_config: a dict with at least "method" (a key of METHODS) and "name".

Each example appends one row to results/<exp>/runs.jsonl (prediction, score,
timings, peak VRAM, selected token ranges, evidence recall, status, code commit).
The row id is <hash of the method config>:<example id>, so a rerun of the same
command skips what is done, and a changed method config is a different method.
Rows with status "failed" (an error or out-of-memory on that example) count as done
unless retry_failed is set; a retry appends a new row, and readers take the last
row per id. A lost GPU is never a row: the run writes interrupted_<hash>.json and
exits with status 75 (EXIT_GPU_LOST); rerun the same command once the GPU is back.

Pilot mode runs the first `pilot_n` (5) pending examples and prints seconds per
example, peak VRAM and the projected time for the whole cell.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from pcm.eval import metrics
from pcm.runlog import JsonlLog, atomic_write_json
from pcm.tasks.common import REPO_ROOT, config_hash, get_tokenizer, iter_jsonl, load_task_config, repo_path

ROW_SCHEMA = 1
GIB = 1024**3


class DoesNotFit(Exception):
    """The prompt is longer than the method allows (recorded as status "skipped")."""


class DirtyTree(RuntimeError):
    """Code or configuration has uncommitted changes."""


# ---------------------------------------------------------------- examples
def parse_task(task: str) -> tuple[str, str]:
    if ":" not in task:
        raise ValueError(f"task must be '<benchmark>:<name>', got {task!r}")
    bench, name = task.split(":", 1)
    if bench not in ("synth", "babilong", "infbench"):
        raise ValueError(f"unknown benchmark {bench!r}")
    return bench, name


def load_examples(task: str, length: str, n: int | None, seed: int, cfg: dict | None = None) -> list[dict]:
    """The first n examples of a cell, in the harness's common form."""
    cfg = cfg or load_task_config()
    bench, name = parse_task(task)
    tok = get_tokenizer(cfg["tokenizer"])
    if bench == "synth":
        from pcm.tasks.synth import data_file

        path = data_file(name, length, seed, cfg)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: materialise it with scripts/make_data.py")
        out = []
        for row in iter_jsonl(path):
            if n is not None and len(out) >= n:
                break
            out.append({**row, "benchmark": "synth"})
        if n is not None and len(out) < n:
            raise ValueError(f"{path} holds {len(out)} examples, {n} requested")
        return out
    if seed != 0:
        raise ValueError(f"{bench} has one fixed test set; use seed 0")
    if bench == "babilong":
        from pcm.tasks import babilong

        return babilong.load(name, length, tok, cfg, n=n)
    from pcm.tasks import infbench

    if length != "full":
        raise ValueError("InfiniteBench examples have their own lengths; use length 'full'")
    return infbench.load(name, tok, cfg, n=n)


def prompt_for(example: dict, cfg: dict | None = None) -> tuple[str, str, str]:
    """(system, user text before the context, user text after it) for any example."""
    bench = example.get("benchmark", "synth")
    if bench == "synth":
        from pcm.tasks.synth import prompt_parts

        return prompt_parts(example, cfg)
    if bench == "babilong":
        from pcm.tasks.babilong import prompt_parts

        return prompt_parts(example["task"], example["question"], cfg)
    from pcm.tasks.infbench import prompt_parts

    return prompt_parts(example["row"], example["task"], cfg)


# ---------------------------------------------------------------- methods
class Method:
    needs_gpu = False

    def __init__(self, config: dict):
        self.config = config

    def setup(self) -> None:
        pass

    def answer(self, example: dict, prompt: tuple[str, str, str]) -> dict:
        """Return {"prediction", "prompt_tokens", "selected_spans", "selected_pages", "timings"}."""
        raise NotImplementedError


class EchoAnswer(Method):
    """Harness tests only: answers with the gold answer. Never a result."""

    def answer(self, example, prompt):
        t0 = time.perf_counter()
        gold = example.get("label", example["answers"])
        pred = gold[0] if example.get("benchmark") == "infbench" and example["task"] == "longbook_choice_eng" else ", ".join(map(str, example["answers"]))
        return {"prediction": pred, "prompt_tokens": None, "selected_spans": [[0, example["length_tokens"]]],
                "selected_pages": None, "timings": {"total_s": time.perf_counter() - t0}}


class Native(Method):
    """The reader on the whole prompt: prefill in blocks, greedy decoding, thinking off.

    config: reader (assets key), dtype (bfloat16), block (2048), max_prompt_tokens
    (prompts longer than this are skipped, not truncated), min_free_vram_gib.
    """

    needs_gpu = True

    def setup(self):
        import torch

        from pcm.models import load_reader, stop_ids

        dtype = getattr(torch, self.config.get("dtype", "bfloat16"))
        self.model, self.tokenizer = load_reader(self.config["reader"], dtype=dtype)
        self.tokenizer.model_max_length = 10**12
        self.stops = stop_ids(self.tokenizer)

    def answer(self, example, prompt):
        import torch

        from pcm.models import chat_segments, encode_text
        from pcm.prefill import generate_greedy

        system, before, after = prompt
        chat = chat_segments(self.tokenizer, system, before, after, example["context"])
        ctx_ids = encode_text(self.tokenizer, chat.context)
        if len(ctx_ids) != example["length_tokens"]:
            raise RuntimeError(f"context tokenizes to {len(ctx_ids)} tokens, the data says {example['length_tokens']}")
        ids = torch.cat([encode_text(self.tokenizer, chat.prefix), ctx_ids, encode_text(self.tokenizer, chat.suffix)])
        limit = int(self.config.get("max_prompt_tokens", 32768))
        if len(ids) > limit:
            raise DoesNotFit(f"prompt has {len(ids)} tokens, method allows {limit}")
        timings: dict = {}
        out = generate_greedy(self.model, None, ids, int(example["max_new_tokens"]), self.stops, int(self.config.get("block", 2048)), timings=timings)
        pred = self.tokenizer.decode(out, skip_special_tokens=True).strip()
        return {"prediction": pred, "prompt_tokens": int(len(ids)), "selected_spans": [[0, int(len(ctx_ids))]],
                "selected_pages": None, "timings": timings}


METHODS = {"native": Native, "echo_answer": EchoAnswer}


def make_method(config: dict) -> Method:
    if config.get("method") not in METHODS:
        raise ValueError(f"unknown method {config.get('method')!r}; known: {sorted(METHODS)}")
    return METHODS[config["method"]](config)


# ---------------------------------------------------------------- run
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_example_fields(ex: dict) -> dict:
    return {
        "example_id": ex["id"],
        "benchmark": ex.get("benchmark", "synth"),
        "family": ex.get("family"),
        "index": ex.get("index"),
        "length_tokens": ex.get("length_tokens"),
        "depth_decile": ex.get("depth_decile"),
        "answers": ex.get("answers"),
    }


def run(
    method_config: dict,
    task: str,
    length: str,
    n: int | None,
    seed: int = 0,
    exp: str = "scratch",
    pilot: bool = False,
    pilot_n: int = 5,
    retry_failed: bool = False,
    allow_dirty: bool = False,
    results_root: Path | None = None,
    examples: list[dict] | None = None,
    cfg: dict | None = None,
    log=print,
) -> dict:
    """Run one method on one cell. Returns a summary (with the pilot projection in pilot mode)."""
    from pcm.env import collect_env
    from pcm.health import EXIT_GPU_LOST, GpuTelemetry, classify_failure, interruption_record, preflight

    cfg = cfg or load_task_config()
    exp_dir = Path(results_root or REPO_ROOT / "results") / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    if env["git_dirty"] and not allow_dirty:
        raise DirtyTree("uncommitted code or configuration: commit first so every row carries a clean commit (or pass allow_dirty)")
    atomic_write_json(exp_dir / "env.json", env)
    JsonlLog(exp_dir / "env_history.jsonl", id_field="timestamp_utc").append({**env, "task": task, "length": length})

    mhash = config_hash(method_config)
    atomic_write_json(exp_dir / "methods" / f"{mhash}.json", method_config)
    if examples is None:
        examples = load_examples(task, length, n, seed, cfg)
    elif n is not None:
        examples = examples[:n]
    runs = JsonlLog(exp_dir / "runs.jsonl")

    def counts_as_done(row: dict) -> bool:
        return row.get("status") in ("ok", "skipped") or (row.get("status") == "failed" and not retry_failed)

    done = runs.done_ids(where=counts_as_done)
    pending = [e for e in examples if f"{mhash}:{e['id']}" not in done]
    todo = pending[:pilot_n] if pilot else pending
    log(f"[{exp}] {method_config.get('name')} ({mhash}) {task} {length} seed {seed}: "
        f"{len(examples)} examples, {len(examples) - len(pending)} done, running {len(todo)}{' (pilot)' if pilot else ''}")

    method = make_method(method_config)
    gpu = method.needs_gpu
    if gpu:
        import torch

        pf = preflight(min_free_vram_gib=float(method_config.get("min_free_vram_gib", 0.0)),
                       min_free_ram_gib=float(method_config.get("min_free_ram_gib", 2.0)),
                       min_free_disk_gib=1.0, disk_path=exp_dir)
        log(f"preflight: {pf}")
    telemetry = GpuTelemetry(REPO_ROOT / "logs" / exp / "gpu_telemetry.csv") if gpu else nullcontext()
    seconds: list[float] = []
    peaks: list[float] = []
    status_counts = {"ok": 0, "failed": 0, "skipped": 0}
    last_done = None
    with telemetry:
        if todo:
            method.setup()
        for ex in todo:
            row = {"id": f"{mhash}:{ex['id']}", "schema": ROW_SCHEMA, "exp": exp, "method": method_config.get("name"),
                   "method_hash": mhash, "task": task, "length": length, "seed": seed, **_row_example_fields(ex),
                   "pilot": pilot, "git_commit": env["git_commit"], "git_dirty": env["git_dirty"]}
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                out = method.answer(ex, prompt_for(ex, cfg))
                total = time.perf_counter() - t0
                sc = metrics.score(out["prediction"], ex)
                row.update({
                    "status": "ok", "prediction": out["prediction"], "score": sc, "correct": sc == 1.0,
                    "prompt_tokens": out.get("prompt_tokens"), "selected_pages": out.get("selected_pages"),
                    "selected_spans": out.get("selected_spans"),
                    "evidence_recall": metrics.evidence_item_recall(ex.get("evidence") or [], out.get("selected_spans") or []),
                    "timings": {**out.get("timings", {}), "total_s": total},
                })
                seconds.append(total)
            except DoesNotFit as e:
                row.update({"status": "skipped", "reason": str(e), "score": None})
            except Exception as e:  # noqa: BLE001 - every failure is classified and recorded
                kind = classify_failure(e)
                if kind == "gpu_lost":
                    atomic_write_json(exp_dir / f"interrupted_{mhash}.json", {**interruption_record(e, last_done), "task": task, "length": length})
                    log(f"GPU lost during {ex['id']}: {e}. Nothing was recorded for it. Rerun the same command once the GPU is back.")
                    raise SystemExit(EXIT_GPU_LOST)
                row.update({"status": "failed", "failure_kind": kind, "error": f"{type(e).__name__}: {str(e)[:500]}", "score": None})
                if kind == "oom" and gpu:
                    torch.cuda.empty_cache()
            if gpu:
                row["peak_vram_gib"] = torch.cuda.max_memory_allocated() / GIB
                row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
                peaks.append(row["peak_vram_gib"])
            row["timestamp_utc"] = _now()
            runs.append(row)
            status_counts[row["status"]] += 1
            last_done = ex["id"]

    remaining = len(pending) - len(todo)
    sec = sum(seconds) / len(seconds) if seconds else None
    summary = {"exp": exp, "method": method_config.get("name"), "method_hash": mhash, "task": task, "length": length,
               "seed": seed, "examples": len(examples), "ran": len(todo), **status_counts, "remaining": remaining,
               "seconds_per_example": sec, "peak_vram_gib": max(peaks) if peaks else None}
    if pilot:
        summary["projected_s_for_cell"] = sec * (len(pending)) if sec is not None else None
        summary["projected_s_remaining"] = sec * remaining if sec is not None else None
        proj = summary["projected_s_for_cell"]
        log(f"pilot: {len(todo)} examples, {sec:.2f} s/example, peak VRAM {summary['peak_vram_gib'] or 0:.2f} GiB; "
            f"projected {proj / 3600:.2f} h for the {len(pending)} pending examples of this cell" if sec is not None else "pilot: no successful example")
    return summary


def _load_method_config(spec: str) -> dict:
    p = Path(spec)
    if p.suffix in (".yaml", ".yml", ".json") and repo_path(p).is_file():
        import yaml

        return yaml.safe_load(repo_path(p).read_text(encoding="utf-8"))
    return json.loads(spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one method on one task cell; resumable.")
    ap.add_argument("--method", required=True, help="method config: a YAML/JSON file or inline JSON")
    ap.add_argument("--task", required=True)
    ap.add_argument("--length", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    summary = run(_load_method_config(args.method), args.task, args.length, args.n, args.seed, args.exp,
                  pilot=args.pilot, retry_failed=args.retry_failed, allow_dirty=args.allow_dirty)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
