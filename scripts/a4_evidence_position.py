"""Where the gold evidence sits in the prompt, how many distractors were kept, and accuracy by both.

    python scripts/a4_evidence_position.py --task infbench:kv_retrieval
    python scripts/a4_evidence_position.py --task synth:S2_multikey --lengths 32k 64k 128k 256k

Written to explain two A4 results: on kv_retrieval BM25 (M2) scores above the oracle (M3), and on
S2 the unit methods score higher at 256K than at 32K. For each example the document is cut into
index units again (CPU only, the same cut as the runs); for each stored row of the A4 unit methods
the kept units are read back from the row's bitmap. Recorded per row: where the gold units fall in
the kept text (0 = start, 1 = end) and the kept tokens after them; how many kept units are a prefix
of the document; how many distractor items (synthetic tasks) lie wholly inside the kept units;
and, for BM25, how many units score above zero.
Output: results/A4_baselines/evidence_position_<task>.json.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.chunking import cut_document  # noqa: E402
from pcm.eval import metrics as M  # noqa: E402
from pcm.eval import runner as R  # noqa: E402
from pcm.index import select as S  # noqa: E402
from pcm.index.bm25 import BM25  # noqa: E402
from pcm.models import chat_segments  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json  # noqa: E402
from pcm.tasks.common import get_tokenizer, load_task_config  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from a4_summary import current_hashes  # noqa: E402
from run_a4 import cell_examples  # noqa: E402

BINS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0001]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="infbench:kv_retrieval")
    ap.add_argument("--lengths", nargs="+", default=["full"])
    args = ap.parse_args()
    tcfg = load_task_config()
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "a4_baselines.yaml").read_text(encoding="utf-8"))
    mcfg = tcfg["metrics"]
    current = current_hashes(cfg)
    methods = {name: m for name, m in cfg["methods"].items() if m["method"] in ("retrieve", "oracle")}
    unit, snap = int(cfg["units"]["unit"]), int(cfg["units"]["snap"])
    bm25_cfg = cfg["methods"]["m2-bm25-1.7B"]["bm25"]

    rows = defaultdict(dict)  # (method, length) -> example id -> row
    for exp in ("E3_retrieval", "E3o_oracle"):
        last = {}
        for r in JsonlLog(REPO_ROOT / "results" / exp / "runs.jsonl").rows():
            last[r["id"]] = r
        for r in last.values():
            if r["task"] == args.task and r["length"] in args.lengths and r["status"] == "ok" and r["method"] in methods \
                    and r["method_hash"] in current[r["method"]]:
                rows[(r["method"], r["length"])][r["example_id"]] = r

    def acc(rs):
        return M.bootstrap_ci([x["score"] for x in rs], mcfg["bootstrap_resamples"], mcfg["bootstrap_seed"], mcfg["ci"]) if rs else None

    tok = get_tokenizer(tcfg["tokenizer"])
    out = {"task": args.task, "unit": unit, "snap": snap, "bins": BINS[:-1] + [1.0], "lengths": {}}
    for L in args.lengths:
        examples = cell_examples({"task": args.task, "length": L}, cfg["n"], cfg, tcfg)
        per_method = defaultdict(list)
        docs = []
        for ex in examples:
            chat = chat_segments(tok, *R.prompt_for(ex, tcfg), ex["context"])
            units = cut_document(tok, chat.context, unit, snap)
            gold = sorted(set(u for it in S.evidence_units(ex.get("evidence") or [], units.bounds) for u in it))
            distract = S.evidence_units(ex.get("distractors") or [], units.bounds)
            bm = BM25.from_texts(units.texts(ex["context"]), k1=bm25_cfg["k1"], b=bm25_cfg["b"], epsilon=bm25_cfg["epsilon"]).scores(R.query_text(ex))
            docs.append({"example_id": ex["id"], "n_units": units.n_units, "gold_units": len(gold), "distractors": len(distract),
                         "bm25_units_above_zero": int(np.sum(bm > 0))})
            sizes = units.sizes
            for (m, length), by_id in rows.items():
                r = by_id.get(ex["id"]) if length == L else None
                if r is None or not (r.get("selection") or {}).get("units_b64"):
                    continue
                if r["selection"]["n_units"] != units.n_units:
                    raise RuntimeError(f"{m} {ex['id']}: the row has {r['selection']['n_units']} units, the cut gives {units.n_units}")
                kept = S.unpack_units(r["selection"]["units_b64"], units.n_units)
                kept_set = set(int(u) for u in kept)
                tok_before = np.cumsum(np.concatenate([[0], sizes[kept]]))  # kept tokens before each kept unit
                total = int(tok_before[-1])
                gold_kept = [u for u in gold if u in kept_set]
                rec = {"example_id": ex["id"], "score": r["score"], "evidence_recall": r.get("evidence_recall"), "kept_units": len(kept),
                       "prefix_units": int(np.argmax(np.diff(np.concatenate([kept, [10**9]])) != 1) + 1) if len(kept) and kept[0] == 0 else 0,
                       "gold_kept": len(gold_kept) == len(gold) and len(gold) > 0,
                       "distractors_kept": sum(1 for it in distract if it and all(u in kept_set for u in it))}
                if gold_kept:
                    first, last_ = int(np.searchsorted(kept, gold_kept[0])), int(np.searchsorted(kept, gold_kept[-1]))
                    rec["position"] = float(tok_before[first] / total)
                    rec["tokens_after"] = int(total - tok_before[last_ + 1])
                per_method[m].append(rec)

        res = {"examples": len(examples), "methods": {}, "documents": docs}
        for m, recs in sorted(per_method.items()):
            kept_all = [x for x in recs if x["gold_kept"]]
            bins = {}
            for lo, hi in zip(BINS[:-1], BINS[1:]):
                b = [x for x in kept_all if lo <= x["position"] < hi]
                bins[f"[{lo:.2f}, {min(hi, 1.0):.2f}]"] = {"n": len(b), "accuracy": acc(b)}
            by_d = {}
            for d in sorted(set(x["distractors_kept"] for x in recs)):
                b = [x for x in recs if x["distractors_kept"] == d]
                by_d[str(d)] = {"n": len(b), "accuracy": acc(b)}
            res["methods"][m] = {
                "rows": len(recs), "accuracy": acc(recs), "gold_kept": len(kept_all), "accuracy_gold_kept": acc(kept_all),
                "accuracy_gold_missing": acc([x for x in recs if not x["gold_kept"]]),
                "position_median": float(np.median([x["position"] for x in kept_all])) if kept_all else None,
                "tokens_after_median": float(np.median([x["tokens_after"] for x in kept_all])) if kept_all else None,
                "prefix_units_median": float(np.median([x["prefix_units"] for x in recs])),
                "distractors_kept_mean": float(np.mean([x["distractors_kept"] for x in recs])),
                "accuracy_by_position": bins, "accuracy_by_distractors_kept": by_d, "rows_detail": recs}
        bz = [d["bm25_units_above_zero"] for d in docs]
        res["bm25_units_above_zero"] = {"median": float(np.median(bz)), "max": int(max(bz)), "n_units_median": float(np.median([d["n_units"] for d in docs]))}
        out["lengths"][L] = res
        atomic_write_json(REPO_ROOT / "results" / "A4_baselines" / f"evidence_position_{args.task.split(':')[1]}.json", out)
        print(f"== {args.task} {L}: bm25 units above zero {res['bm25_units_above_zero']}")
        for m, v in res["methods"].items():
            a, g = v["accuracy"], v["accuracy_gold_kept"]
            print(f"{m}: acc {a['mean']:.3f} n={v['rows']}; gold kept {v['gold_kept']} (acc {g['mean'] if g else None}); median position {v['position_median']}, "
                  f"tokens after {v['tokens_after_median']}, prefix units {v['prefix_units_median']}, distractors kept {v['distractors_kept_mean']:.2f}")
            print("    by position:", {k: (b["n"], round(b["accuracy"]["mean"], 3) if b["accuracy"] else None) for k, b in v["accuracy_by_position"].items()})
            print("    by distractors kept:", {k: (b["n"], round(b["accuracy"]["mean"], 3) if b["accuracy"] else None) for k, b in v["accuracy_by_distractors_kept"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
