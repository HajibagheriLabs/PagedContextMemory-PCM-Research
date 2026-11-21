"""Entities and ENT edges for E5's documents, computed on the CPU ahead of the run (section 4.7 rule 10).

    python scripts/a5_ingest_entities.py --extractor spacy --workers 3
    python scripts/a5_ingest_entities.py --extractor spacy --workers 2 --tasks babilong:qa2 --lengths 256k

spaCy's NER takes about 35 s per 256K-token document on one core. Done inside the run, it would hold the
GPU idle; done here, in worker processes, it can run beside E4. Each document is cut into index units
exactly as the runner cuts it, keyed by the hash of its token ids, and its entities and ENT edges are
written to the same per-document cache the runner reads (data/cache/index/.../graph/ent-<extractor>-c<cap>).
Documents already cached are skipped, so the script can be stopped and restarted.

Default: the documents of every E5 cell whose graph variant uses the given extractor (configs/a5_pillars.yaml).
Progress goes to logs/A5/ingest_entities_<extractor>.log.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

_W: dict = {}


def _init(extractor: str, cap: int, unit: int, snap: int, cache_root: str) -> None:
    from pcm.hub import model_path
    from pcm.index import graph as G
    from pcm.index.embed import embedder_tag
    from pcm.tasks.common import get_tokenizer, load_task_config, tokenizer_tag

    tcfg = load_task_config()
    tok = get_tokenizer(tcfg["tokenizer"])
    root = Path(cache_root) / tokenizer_tag(tok) / embedder_tag(model_path("embedder").name, "bfloat16") / f"u{unit}s{snap}"
    _W.update({"tok": tok, "tcfg": tcfg, "cache": G.GraphCache(root), "name": G.ent_name(extractor, cap), "extractor": extractor,
               "cap": cap, "unit": unit, "snap": snap})
    if extractor == "spacy":
        G.spacy_model()


def _one(ex: dict) -> dict:
    from pcm.chunking import cut_document
    from pcm.eval import runner as R
    from pcm.index import graph as G
    from pcm.index.ingest import doc_hash
    from pcm.models import chat_segments

    t0 = time.perf_counter()
    chat = chat_segments(_W["tok"], *R.prompt_for(ex, _W["tcfg"]), ex["context"])
    units = cut_document(_W["tok"], chat.context, _W["unit"], _W["snap"])
    if units.n_tokens != ex["length_tokens"]:
        raise RuntimeError(f"{ex['id']}: context tokenizes to {units.n_tokens} tokens, the data says {ex['length_tokens']}")
    key = doc_hash(units.ids)
    if _W["cache"].path(_W["name"], key).is_file():
        return {"id": ex["id"], "doc": key, "cached": True, "s": time.perf_counter() - t0}
    t1 = time.perf_counter()
    ents = G.extract(units.texts(ex["context"]), _W["extractor"])
    edges = G.ent_edges(ents, _W["cap"])
    size = _W["cache"].save(_W["name"], key, G.pack_entities(ents, edges))
    return {"id": ex["id"], "doc": key, "cached": False, "s": time.perf_counter() - t0, "extract_s": time.perf_counter() - t1,
            "units": units.n_units, "entities": len(ents.vocab), "edges": len(edges[0]), "bytes": size}


def main() -> int:
    import yaml

    from pcm.runlog import JsonlLog
    from pcm.tasks.common import load_task_config
    from run_a4 import cell_examples
    from run_a5 import expand_tasks, lengths_for

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractor", default="spacy", choices=["default", "spacy"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--lengths", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "a5_pillars.yaml").read_text(encoding="utf-8"))
    tcfg = load_task_config()
    cells = []
    for entry in cfg["e5"]["plan"]:
        uses = [v for v in entry["variants"] if (cfg["e5"]["variants"].get(v) or {}).get("extractor", "default") == args.extractor
                and (cfg["e5"]["variants"].get(v) or {}).get("traversal") is not None]
        if not uses:
            continue
        for t in expand_tasks(cfg, entry["tasks"]):
            for L in entry["lengths"]:
                for LL in lengths_for(t, [L]):
                    if (t, LL) not in cells and (not args.tasks or t in args.tasks) and (not args.lengths or LL in args.lengths):
                        cells.append((t, LL))
    log_path = REPO_ROOT / "logs" / "A5" / f"ingest_entities_{args.extractor}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = JsonlLog(log_path.with_suffix(".jsonl"), id_field="id")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"{args.extractor} entities for {len(cells)} cells with {args.workers} workers: {cells}")
    unit, snap = int(cfg["pcm"]["unit"]), int(cfg["pcm"]["snap"])
    cache_root = str(REPO_ROOT / cfg["pcm"]["index_cache"])
    cap = int(cfg["graph"]["ent_cap"])
    t0, done, built = time.perf_counter(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.extractor, cap, unit, snap, cache_root)) as pool:
        for t, L in cells:
            examples = cell_examples({"task": t, "length": L}, cfg["n"], cfg, tcfg)
            futs = [pool.submit(_one, ex) for ex in examples]
            for f in as_completed(futs):
                r = f.result()
                done += 1
                built += 0 if r["cached"] else 1
                if not r["cached"]:
                    rec.append({**r, "task": t, "length": L, "extractor": args.extractor})
            log(f"{t} {L}: {len(examples)} documents; {done} done ({built} built) in {(time.perf_counter() - t0) / 60:.1f} min")
    log(f"finished: {done} documents, {built} built, {(time.perf_counter() - t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
