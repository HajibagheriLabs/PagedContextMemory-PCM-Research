"""Resolve every model, dataset and reference repository listed in configs/assets.yaml.

Local-first: an asset already in the Hugging Face hub cache is used as is; the
network is touched only for what is missing (and never with --offline). Each
asset is checked for completeness and given a quick read test. Writes
reports/assets.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.hub import (  # noqa: E402
    hub_cache_dir,
    load_assets,
    missing_files,
    resolve_snapshot,
    snapshot_size,
)

GIB = 1024**3
MIB = 1024**2


def weight_files(snapshot: Path) -> list[str]:
    return sorted(p.name for p in snapshot.glob("*.safetensors"))


def fetch_model(key: str, repo: str, required: list[str], allow_download: bool) -> dict:
    snap, source = resolve_snapshot(repo, "model", allow_download=allow_download)
    missing = missing_files(snap, required)
    weights = weight_files(snap)
    if not weights:
        missing.append("*.safetensors")
    index = snap / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        missing += missing_files(snap, shards)
    n, size = snapshot_size(snap)
    return {
        "key": key,
        "repo": repo,
        "type": "model",
        "revision": snap.name,
        "source": source,
        "files": n,
        "bytes": size,
        "weights": weights,
        "missing": missing,
    }


def read_json_head(path: Path) -> dict:
    """Shape of a BABILong json file: number of rows and the field names."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"rows": len(data), "fields": sorted(data[0].keys()) if data else []}
    if isinstance(data, dict):
        # column-oriented layout: field -> {row index -> value}
        first = next(iter(data.values()))
        return {"rows": len(first), "fields": sorted(data.keys())}
    return {"rows": None, "fields": []}


def fetch_babilong(spec: dict, allow_download: bool) -> dict:
    lengths, tasks = spec["lengths"], spec["tasks"]
    patterns = ["README.md"] + [f"data/{t}/{n}.json" for t in tasks for n in lengths]
    snap, source = resolve_snapshot(spec["repo"], "dataset", patterns, allow_download)
    required = [f"data/{t}/{n}.json" for t in tasks for n in lengths]
    missing = missing_files(snap, required)
    per_length = {}
    for n in lengths:
        files = [snap / "data" / t / f"{n}.json" for t in tasks]
        per_length[n] = sum(f.stat().st_size for f in files if f.is_file())
    extra = sorted(
        {p.stem for p in (snap / "data").glob("qa*/*.json")} - set(lengths),
        key=lambda s: s,
    )
    head = read_json_head(snap / "data" / tasks[0] / f"{lengths[0]}.json") if not missing else {}
    n_files, size = snapshot_size(snap)
    return {
        "key": "babilong",
        "repo": spec["repo"],
        "type": "dataset",
        "revision": snap.name,
        "source": source,
        "layout": "lengths are configs, tasks are splits (dataset card); files data/<task>/<length>.json",
        "files": n_files,
        "bytes": size,
        "bytes_required_lengths": sum(per_length.values()),
        "bytes_per_length": per_length,
        "other_lengths_present": extra,
        "sample_shape": head,
        "missing": missing,
    }


def fetch_babilong_train(spec: dict, allow_download: bool) -> dict:
    snap, source = resolve_snapshot(spec["repo"], "dataset", allow_download=allow_download)
    files = sorted((snap / "data").glob("qa*/*.json"))
    lengths = sorted({p.stem for p in files})
    tasks = sorted({p.parent.name for p in files}, key=lambda s: int(s[2:]))
    n_files, size = snapshot_size(snap)
    return {
        "key": "babilong_train",
        "repo": spec["repo"],
        "type": "dataset",
        "revision": snap.name,
        "source": source,
        "files": n_files,
        "bytes": size,
        "tasks_present": tasks,
        "lengths_present": lengths,
        "missing": [] if files else ["data/qa*/*.json"],
    }


def infbench_first_row(path: Path) -> dict:
    """First row of a jsonl file, read as a stream with the dataset card's Features spec."""
    from datasets import Features, Sequence, Value, load_dataset

    ft = Features(
        {
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string")),
        }
    )
    ds = load_dataset("json", data_files=str(path), features=ft, split="train", streaming=True)
    row = next(iter(ds))
    return {
        "id": row["id"],
        "context_chars": len(row["context"]),
        "input": row["input"][:120],
        "answer": row["answer"],
        "n_options": len(row["options"] or []),
    }


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def fetch_infbench(spec: dict, allow_download: bool) -> dict:
    patterns = ["README.md"] + list(spec["files"])
    snap, source = resolve_snapshot(spec["repo"], "dataset", patterns, allow_download)
    missing = missing_files(snap, spec["files"])
    per_file = {}
    for name in spec["files"]:
        p = snap / name
        if p.is_file():
            per_file[name] = {
                "bytes": p.stat().st_size,
                "rows": count_lines(p),
                "first_row": infbench_first_row(p),
            }
    n_files, size = snapshot_size(snap)
    return {
        "key": "infbench",
        "repo": spec["repo"],
        "type": "dataset",
        "revision": snap.name,
        "source": source,
        "files": n_files,
        "bytes": size,
        "bytes_required_files": sum(v["bytes"] for v in per_file.values()),
        "per_file": per_file,
        "missing": missing,
    }


def fetch_pg19(spec: dict, allow_download: bool) -> dict:
    from pcm.tasks.pg19 import split_info, stream_books

    patterns = ["README.md"] + [f"data/{s}-*.parquet" for s in spec["splits"]]
    snap, source = resolve_snapshot(spec["repo"], "dataset", patterns, allow_download)
    missing = [s for s in spec["splits"] if not list((snap / "data").glob(f"{s}-*.parquet"))]
    splits = {}
    for s in spec["splits"]:
        if s in missing:
            continue
        info = split_info(s)
        book = next(stream_books(s))
        text_col = "text" if "text" in book else max(book, key=lambda k: len(str(book[k])))
        info["bytes"] = sum(p.stat().st_size for p in (snap / "data").glob(f"{s}-*.parquet"))
        info["first_book"] = {
            k: (v if k != text_col else f"<{len(v)} chars>") for k, v in book.items()
        }
        splits[s] = info
    train_present = bool(list((snap / "data").glob("train-*.parquet")))
    n_files, size = snapshot_size(snap)
    return {
        "key": "pg19",
        "repo": spec["repo"],
        "type": "dataset",
        "revision": snap.name,
        "source": source,
        "files": n_files,
        "bytes": size,
        "splits": splits,
        "train_split_on_disk": train_present,
        "missing": missing,
    }


def reference_repo(name: str, url: str, allow_download: bool) -> dict:
    dest = REPO_ROOT / "third_party" / name
    source = "present"
    if not (dest / ".git").exists():
        if not allow_download:
            return {"name": name, "url": url, "missing": True}
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
        source = "cloned"

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(dest), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "name": name,
        "url": url,
        "origin": git("remote", "get-url", "origin"),
        "commit": git("rev-parse", "HEAD"),
        "commit_date": git("log", "-1", "--format=%cs"),
        "source": source,
        "missing": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true", help="never touch the network")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "assets.json")
    args = ap.parse_args()
    allow = not args.offline
    cfg = load_assets()

    report: dict = {"hub_cache": str(hub_cache_dir()), "models": [], "datasets": [], "reference_repos": []}
    for key, repo in cfg["models"].items():
        report["models"].append(fetch_model(key, repo, cfg["model_required_files"], allow))
    ds = cfg["datasets"]
    report["datasets"].append(fetch_babilong(ds["babilong"], allow))
    report["datasets"].append(fetch_babilong_train(ds["babilong_train"], allow))
    report["datasets"].append(fetch_infbench(ds["infbench"], allow))
    report["datasets"].append(fetch_pg19(ds["pg19"], allow))
    for name, url in cfg["reference_repos"].items():
        report["reference_repos"].append(reference_repo(name, url, allow))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    problems = []
    print(f"hub cache: {report['hub_cache']}")
    for item in report["models"] + report["datasets"]:
        size = item["bytes"]
        size_s = f"{size / GIB:.2f} GiB" if size >= GIB else f"{size / MIB:.1f} MiB"
        status = "OK" if not item["missing"] else f"MISSING {item['missing']}"
        print(f"{item['type']:8s} {item['repo']:42s} {size_s:>10s}  {item['source']:8s} {status}")
        if item["missing"]:
            problems.append(item["repo"])
    for r in report["reference_repos"]:
        status = "MISSING" if r["missing"] else f"{r['commit'][:10]} {r['commit_date']} ({r['source']})"
        print(f"repo     {r['name']:42s} {status}")
        if r["missing"]:
            problems.append(r["name"])
    print(f"\nwrote {args.out}")
    if problems:
        print(f"INCOMPLETE: {problems}")
        sys.exit(1)


if __name__ == "__main__":
    main()
