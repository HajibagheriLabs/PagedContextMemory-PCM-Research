"""Numbers, tables and figures for reports/A8_local.md.

    python scripts/a8_summary.py

Assembles E8's comparison (configs/a8_local.yaml, `series`; D-052) from the rows of E0, E1, E2, E3, E3o, E6 and E8:
M0, M1, M2 hybrid, M3 (and the tight oracle, supplementary), P-T and P-KV on the same example ids. Reads H1 and Gate G3
as fixed in D-053. Writes results/A8_local/summary.json, tables.md and figures/*.png. Every table gives n.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from a4_summary import chance, current_hashes, family_ci  # noqa: E402
from pcm.eval import metrics as M  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json, atomic_write_text  # noqa: E402
from pcm.tasks.common import config_hash, get_tokenizer, load_task_config  # noqa: E402
from run_a8 import load_config  # noqa: E402

OUT = REPO_ROOT / "results" / "A8_local"
FIG = OUT / "figures"
GRID = ["8k", "16k", "32k", "64k", "128k", "256k"]
LONG = ["32k", "64k", "128k", "256k"]
GIB = 1024**3
SERIES_ORDER = ["M0", "M1", "M2", "M3", "P-T", "P-KV"]
# Categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision >= 19.6 on the light surface);
# every series also has its own marker and line style, and every figure has a table in the report.
STYLE = {"P-T": ("#2a78d6", "o", "-"), "P-KV": ("#eb6834", "s", "-"), "M2": ("#1baf7a", "^", "-"), "M3": ("#eda100", "D", "-"),
         "M1": ("#e87ba4", "v", "-"), "M0": ("#008300", "<", "--"), "M3-tight": ("#6250d6", "d", ":")}


def rows_of(exp: str, name: str = "runs.jsonl") -> list[dict]:
    p = REPO_ROOT / "results" / exp / name
    if not p.is_file():
        return []
    last: dict = {}
    for r in JsonlLog(p).rows():
        last[r.get("id")] = r
    return list(last.values())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else None


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


def pct(c, d=1):
    if not c or c.get("mean") is None:
        return "–"
    return f"{100 * c['mean']:.{d}f} [{100 * c['lo']:.{d}f}, {100 * c['hi']:.{d}f}]"


def p0(c, d=1):
    return "–" if not c or c.get("mean") is None else f"{100 * c['mean']:.{d}f}"


def dpct(x, d=1):
    if not x or x.get("diff") is None:
        return "–"
    return f"{100 * x['diff']:+.{d}f} [{100 * x['lo']:+.{d}f}, {100 * x['hi']:+.{d}f}]"


def f2(x, k=2):
    return "–" if x is None else f"{x:.{k}f}"


# ---------------------------------------------------------------- BABILong: the facts an answer rests on (qa1, qa2)
MOVE = re.compile(r"^(\w+) (?:moved|went|journeyed|travelled|went back) to the (\w+)\.?$")
GET = re.compile(r"^(\w+) (?:got|took|picked up|grabbed) the (\w+)(?: there)?\.?$")
DROP = re.compile(r"^(\w+) (?:dropped|discarded|put down|left) the (\w+)(?: there)?\.?$")


def supporting_facts(task: str, facts: list[str], question: str, target: str):
    """Indices of the facts the answer rests on, by replaying the bAbI story (qa1: the person's last move; qa2: the
    object's last pick-up or drop and the holder's move that places it), or None if the replay does not give the target."""
    q = question.strip().rstrip("?").split()
    if not q:
        return None
    if task == "qa1":
        who = q[-1]
        last = None
        for i, f in enumerate(facts):
            m = MOVE.match(f.strip())
            if m and m.group(1) == who:
                last = (i, m.group(2))
        return [last[0]] if last and last[1] == target else None
    if task == "qa2":
        obj = q[-1]
        loc, last_move = {}, {}
        holder, ev, place, place_fact = None, None, None, None
        for i, f in enumerate(facts):
            f = f.strip()
            m = MOVE.match(f)
            if m:
                loc[m.group(1)], last_move[m.group(1)] = m.group(2), i
                continue
            m = GET.match(f)
            if m and m.group(2) == obj:
                holder, ev = m.group(1), i
                continue
            m = DROP.match(f)
            if m and m.group(2) == obj:
                holder, ev, place, place_fact = None, i, loc.get(m.group(1)), last_move.get(m.group(1))
        if ev is None:
            return None
        if holder is not None:
            ans, sup = loc.get(holder), [ev, last_move.get(holder)]
        else:
            ans, sup = place, [ev, place_fact]
        return sorted(set(x for x in sup if x is not None)) if ans == target and None not in sup else None
    return None


def main() -> int:
    tcfg = load_task_config()
    cfg = load_config()
    a4cfg = yaml.safe_load((REPO_ROOT / "configs" / "a4_baselines.yaml").read_text(encoding="utf-8"))
    mc = tcfg["metrics"]
    B, SEED, CIL = mc["bootstrap_resamples"], mc["bootstrap_seed"], mc["ci"]
    ser, fams = cfg["series"], cfg["families"]
    retained = [t for ts in fams.values() for t in ts]
    infb = cfg["task_sets"]["infbench"]
    current = current_hashes(a4cfg)
    h_pt, h_m2, h_kv = (config_hash(cfg["_methods"][k]) for k in ("P-T", "M2", "P-KV"))

    def ci(xs):
        return M.bootstrap_ci(list(xs), B, SEED, CIL)

    # ---- rows of every source, last per id, status ok
    cache: dict = {}

    def exp_rows(e):
        if e not in cache:
            cache[e] = rows_of(e)
        return cache[e]

    def wanted(e, method):
        """(exp, method name) -> rows of the configuration E8 compares (A4 methods: current hash only)."""
        out = []
        for r in exp_rows(e):
            if r.get("method") != method:
                continue
            if e in ("E2_dense", "E3_retrieval", "E3o_oracle", "E0_a4budget") and r.get("method_hash") not in current.get(method, {r.get("method_hash")}):
                continue
            if e == "E8_local" and method == "m2-hybrid-1.7B" and r["method_hash"] != h_m2:
                continue
            out.append(r)
        return out

    # series -> (task, length) -> example_id -> row ; later sources override earlier ones for the same id
    S: dict = {k: defaultdict(dict) for k in list(ser) if k != "native"}
    native = defaultdict(dict)
    nat = ser["native"]
    for r in wanted(nat["exp"], nat["method"]):
        if r["status"] == "ok":
            native[(r["task"], r["length"])][r["example_id"]] = r
    for r in wanted(nat["budget_rerun"]["exp"], nat["budget_rerun"]["method"]):
        if r["status"] == "ok" and r["task"] in nat["budget_rerun"]["tasks"]:
            native[(r["task"], r["length"])][r["example_id"]] = r
    all_status: dict = defaultdict(Counter)
    commits: dict = defaultdict(Counter)
    for k, s in ser.items():
        if k == "native":
            continue
        exps = s["exp"] if isinstance(s["exp"], list) else [s["exp"]]
        methods = s["method"] if isinstance(s["method"], list) else [s["method"]]
        if s.get("infbench"):
            methods = methods + [s["infbench"]]
        for e in exps:
            for m in methods:
                for r in wanted(e, m):
                    if r["task"].startswith("infbench:") and s.get("infbench") and m != s["infbench"]:
                        continue
                    all_status[k][r["status"]] += 1
                    if r["status"] != "ok":
                        continue
                    commits[k][f"{e}:{(r.get('git_commit') or '')[:7]}{'+dirty' if r.get('git_dirty') else ''}"] += 1
                    S[k][(r["task"], r["length"])][r["example_id"]] = r
    # P-KV with the bf16 store and the same fix (E6), for the codec's accuracy cost
    kv_bf16 = defaultdict(dict)
    for r in exp_rows("E6_kv"):
        if r.get("method") == "pcm-kv-ape" and r["status"] == "ok":
            kv_bf16[(r["task"], r["length"])][r["example_id"]] = r

    def cell(k, task, L):
        """Rows of series k at (task, L); below the series' first own length, the native run."""
        if k == "native":
            return native.get((task, L), {})
        s = ser[k]
        if L != "full" and GRID.index(L) < GRID.index(s["from"]):
            return native.get((task, L), {})
        return S[k].get((task, L), {})

    def scores(k, task, L):
        return {i: r["score"] for i, r in cell(k, task, L).items() if r.get("score") is not None}

    # ---- reference and curves
    ref = {}
    for fam, tasks in fams.items():
        ref[fam] = family_ci([list(scores("native", t, mc["reference_length"]).values()) for t in tasks], mc)
    curves: dict = {}
    for k in list(S):
        for fam, tasks in fams.items():
            cur = {}
            for L in GRID:
                per = [list(scores(k, t, L).values()) for t in tasks]
                if all(per):
                    c = family_ci(per, mc)
                    c["n_per_task"] = [len(x) for x in per]
                    c["source"] = "native" if GRID.index(L) < GRID.index(ser[k]["from"]) else "method"
                    cur[L] = c
            curves[f"{k}|{fam}"] = cur

    def fam_chance(fam):
        return sum(chance(t, tcfg) for t in fams[fam]) / len(fams[fam])

    ecl: dict = {}
    for k in S:
        for fam in fams:
            acc = {L: c["mean"] for L, c in curves[f"{k}|{fam}"].items()}
            if ref[fam]["mean"] is not None:
                ecl[f"{k}|{fam}"] = M.ecl90(acc, ref[fam]["mean"], fam_chance(fam), GRID, mc["ecl_ratio"], mc["ecl_floor_min"])
    # D-053 (4): readings that do not decide
    alt = {}
    for k in ("P-T", "P-KV"):
        for fam in cfg["h1_families"]:
            acc = {L: c["mean"] for L, c in curves[f"{k}|{fam}"].items()}
            nat32 = family_ci([list(scores("native", t, "32k").values()) for t in fams[fam]], mc)["mean"]
            a = dict(acc)
            if nat32 is not None:
                a["32k"] = nat32
            bypass = M.ecl90(a, ref[fam]["mean"], fam_chance(fam), GRID, mc["ecl_ratio"], mc["ecl_floor_min"])
            own = M.ecl90({L: acc.get(L) for L in LONG if L in acc}, ref[fam]["mean"], fam_chance(fam), LONG, mc["ecl_ratio"], mc["ecl_floor_min"])
            alt[f"{k}|{fam}"] = {"bypass_at_32k": bypass["ecl"], "bypass_stopped": bypass["stopped_at"], "own_curve_from_32k": own["ecl"],
                                 "own_stopped": own["stopped_at"], "native_32k": rnd(nat32)}

    # ---- paired differences, per family and length (first minus second, on the shared ids)
    def paired(ka, kb, fam, L, a_src=None, b_src=None):
        xa, xb = [], []
        for t in fams[fam]:
            sa = a_src(t, L) if a_src else scores(ka, t, L)
            sb = b_src(t, L) if b_src else scores(kb, t, L)
            ids = sorted(set(sa) & set(sb))
            xa += [sa[i] for i in ids]
            xb += [sb[i] for i in ids]
        return M.paired_bootstrap(xa, xb, B, SEED, CIL) if xa else None

    pair_list = [("P-T", "M0"), ("P-T", "M1"), ("P-T", "M2"), ("P-T", "M3"), ("P-T", "M3-tight"), ("P-KV", "P-T")]
    pairs = {}
    for a, b in pair_list:
        for fam in fams:
            for L in LONG:
                p = paired(a, b, fam, L)
                if p is not None:
                    pairs[f"{a}-{b}|{fam}|{L}"] = p
    # int4 store against the bf16 store, KV mode with the same fix (H4 on task accuracy)
    codec = {}
    for L in LONG:
        xa, xb = [], []
        for t in retained:
            sa = scores("P-KV", t, L)
            sb = {i: r["score"] for i, r in kv_bf16.get((t, L), {}).items()}
            ids = sorted(set(sa) & set(sb))
            xa += [sa[i] for i in ids]
            xb += [sb[i] for i in ids]
        if xa:
            codec[L] = {**M.paired_bootstrap(xa, xb, B, SEED, CIL), "same_answer": None}
    same = Counter()
    for L in LONG:
        for t in retained:
            a, b = cell("P-KV", t, L), kv_bf16.get((t, L), {})
            for i in set(a) & set(b):
                same[L, a[i]["prediction"] == b[i]["prediction"]] += 1
    for L in codec:
        tot = same[L, True] + same[L, False]
        codec[L]["same_answer"] = rnd(same[L, True] / tot) if tot else None
        codec[L]["pairs"] = tot
    # pooled gap P-KV - P-T over the 13 retained tasks (as D-051 read G2)
    kv_gap = {}
    for L in LONG:
        xa, xb = [], []
        for t in retained:
            sa, sb = scores("P-KV", t, L), scores("P-T", t, L)
            ids = sorted(set(sa) & set(sb))
            xa += [sa[i] for i in ids]
            xb += [sb[i] for i in ids]
        if xa:
            kv_gap[L] = {**M.paired_bootstrap(xa, xb, B, SEED, CIL), "pairs": len(xa), "kv": rnd(mean(xa)), "text": rnd(mean(xb))}
        # the same 100 examples per task as A6 (the 256K synthetic cells hold 300 here, which shifts the pool towards S2)
        ya, yb = [], []
        for t in retained:
            sa, sb = scores("P-KV", t, L), scores("P-T", t, L)
            ids = sorted(i for i in set(sa) & set(sb) if (cell("P-T", t, L)[i].get("index") or 0) < 100)
            ya += [sa[i] for i in ids]
            yb += [sb[i] for i in ids]
        if ya:
            kv_gap[L]["first100"] = {**M.paired_bootstrap(ya, yb, B, SEED, CIL), "pairs": len(ya), "kv": rnd(mean(ya)), "text": rnd(mean(yb))}

    # ---- InfiniteBench
    inf = {}
    for k in S:
        for t in infb:
            rs = list(cell(k, t, "full").values())
            if rs:
                inf[f"{k}|{t}"] = {"n": len(rs), "accuracy": ci([r["score"] for r in rs]),
                                   "evidence_recall": rnd(mean([r.get("evidence_recall") for r in rs]))}

    # ---- timings: a reused answer takes its timings from the row that produced it
    source = {}
    for e in ("E3_retrieval", "E4_pages", "E5_graph"):
        for r in exp_rows(e):
            if r["id"] in source and not (source[r["id"]].get("timings") or {}).get("reused"):
                continue
            source[r["id"]] = r

    def origin(r):
        n = 0
        while (r.get("timings") or {}).get("reused") and r.get("reused_from") in source and n < 5:
            r = source[r["reused_from"]]
            n += 1
        return r

    def fresh(rs):
        out = [origin(r) for r in rs]
        return [r for r in out if not (r.get("timings") or {}).get("reused")]

    # ---- the 256K table (report section 3)
    store_rows = [r for r in rows_of("E8_local", "store_256k.jsonl") if r.get("status") == "ok"]
    store = {"rows": store_rows, "n": len(store_rows)}
    if store_rows:
        store.update({k: rnd(mean([r[k] for r in store_rows]), 4) for k in ("index_s", "kv_store_s", "kv_question_s", "text_question_s",
                                                                             "peak_allocated_gib", "peak_reserved_gib", "store_allocated_gib",
                                                                             "resident_before_gib")})
        store["store_gib"] = rnd(mean([r["store_bytes"] / GIB for r in store_rows]), 3)
        store["text_store_mib"] = rnd(mean([r["text_store_bytes"] / 2**20 for r in store_rows]), 3)
        store["index_mib"] = rnd(mean([r["index_bytes"] / 2**20 for r in store_rows]), 3)
        store["max_peak_reserved_gib"] = rnd(max(r["peak_reserved_gib"] for r in store_rows), 3)
        store["all_inside_card"] = all(r["inside_card"] for r in store_rows)
        store["same_as_e8_kv"] = dict(Counter(str(r.get("same_as_e8_kv")) for r in store_rows))
        store["same_as_text_row"] = dict(Counter(str(r.get("same_as_text_row")) for r in store_rows))
    ingest_log = []
    idx_dirs = list((REPO_ROOT / "data" / "cache" / "index").glob("*/*/u128s32/ingest.jsonl"))
    for p in idx_dirs:
        for r in JsonlLog(p, id_field="doc").rows():
            if (r.get("tokens") or 0) > 200_000 and r.get("embed_s"):
                ingest_log.append(r)
    t256 = {}
    for k in ("M0", "M2", "M3", "P-T", "P-KV"):
        rs_all = [r for t in retained for r in cell(k, t, "256k").values()]
        if not rs_all:
            continue
        fr = fresh(rs_all)
        acc = {fam: family_ci([list(scores(k, t, "256k").values()) for t in fams[fam]], mc) for fam in fams}
        rec = {fam: rnd(mean([r.get("evidence_recall") for t in fams[fam] for r in cell(k, t, "256k").values()])) for fam in fams}
        q = [(r.get("timings") or {}).get("total_s", 0) - (r.get("timings") or {}).get("ingest_s", 0.0) for r in fr]
        ing = [(r.get("timings") or {}).get("ingest_s") for r in rs_all if (r.get("timings") or {}).get("ingest_s")]
        t256[k] = {"accuracy": acc, "n": {fam: sum(len(scores(k, t, "256k")) for t in fams[fam]) for fam in fams},
                   "evidence_recall": rec, "peak_vram_gib": rnd(max(r.get("peak_vram_gib") or 0 for r in rs_all), 3),
                   "peak_reserved_gib": rnd(max(r.get("peak_reserved_gib") or 0 for r in rs_all), 3),
                   "seconds_per_question": rnd(mean(q), 3), "seconds_n": len(q),
                   "ingest_s_rows": rnd(mean(ing), 3), "ingest_n": len(ing),
                   "kv_encode_s": rnd(mean([(r.get("timings") or {}).get("kv_encode_s") for r in rs_all]), 3),
                   "prompt_tokens": rnd(mean([r.get("prompt_tokens") for r in rs_all]), 1),
                   "failed": all_status[k].get("failed", 0)}
    t256["ingest_log_256k"] = {"documents": len(ingest_log), "embed_s": rnd(mean([r["embed_s"] for r in ingest_log]), 3),
                               "tokens_per_s": rnd(sum(r["tokens"] for r in ingest_log) / sum(r["embed_s"] for r in ingest_log), 1) if ingest_log else None}

    # ---- Gate G3 and H1 (D-053)
    g = cfg["g3"]
    h1fams = cfg["h1_families"]

    def at_least(e, L):
        return e is not None and GRID.index(e) >= GRID.index(L)

    h1_128 = {fam: ecl[f"P-T|{fam}"]["ecl"] for fam in h1fams}
    pt256 = [r for t in retained + infb for r in list(cell("P-T", t, "256k").values())]
    pt_status = Counter()
    for (t, L), d in S["P-T"].items():
        if L == "256k":
            pt_status["ok"] += len(d)
    pt_failed = sum(1 for e in ("E6_kv", "E8_local") for r in exp_rows(e)
                    if r.get("method") in ("pcm-text", "pcm-p512-b8192") and r.get("length") == "256k" and r["status"] != "ok")
    pt_max_res = max((r.get("peak_reserved_gib") or 0) for r in pt256) if pt256 else None
    fit_rows = pt_failed == 0 and pt_max_res is not None and pt_max_res <= g["vram_gib"]
    fit_store = store.get("all_inside_card") if store_rows else None
    part1 = sum(1 for fam in h1fams if at_least(h1_128[fam], g["h1_length"])) >= g["h1_min_families"]
    part2 = bool(fit_rows and (fit_store is not False))
    h1 = sum(1 for fam in h1fams if at_least(ecl[f"P-T|{fam}"]["ecl"], "256k")) >= g["h1_min_families"]
    kv_max_res = max((r.get("peak_reserved_gib") or 0 for t in retained + infb for r in cell("P-KV", t, "256k").values()), default=None)
    gate = {"h1_ecl90_pt": h1_128, "h1_ecl90_m1": {fam: ecl[f"M1|{fam}"]["ecl"] for fam in h1fams},
            "h1_ecl90_pkv": {fam: ecl[f"P-KV|{fam}"]["ecl"] for fam in h1fams},
            "part1_h1_at_128k": part1, "h1_at_256k": h1,
            "part2_pt_rows_256k": {"rows_ok": len(pt256), "failed": pt_failed, "max_peak_reserved_gib": rnd(pt_max_res, 3),
                                   "max_peak_allocated_gib": rnd(max((r.get("peak_vram_gib") or 0) for r in pt256), 3) if pt256 else None},
            "part2_store_check": fit_store, "part2_pcm_at_256k_inside_24gb": part2,
            "pkv_rows_256k_max_peak_reserved_gib": rnd(kv_max_res, 3),
            "g3_passes": bool(part1 and part2)}

    # ---- H5: aggregation (A3's S4 rows; not retained at Gate G0)
    h5 = {}
    for t in ("synth:S4_count", "synth:S4_freq"):
        for L in GRID:
            src = [r for r in exp_rows("E0_floor") if r.get("method") == "native-1.7B" and r["task"] == t and r["length"] == L and r["status"] == "ok"]
            if not src:
                src = [r for r in exp_rows("E1_truncation") if r.get("method") == "m0-tail-1.7B" and r["task"] == t and r["length"] == L and r["status"] == "ok"]
            if src:
                h5[f"{t}|{L}"] = {"method": src[0]["method"], "n": len(src), "accuracy": ci([r["score"] for r in src])}

    # ---- failures of PCM at 256K, with their cause
    tok = get_tokenizer(tcfg["tokenizer"])
    from pcm.tasks import babilong as BL

    facts_of = {}
    for t in [x for x in retained if x.startswith("babilong:")]:
        name = t.split(":")[1]
        p = BL.meta_file(name, "256k", tok, tcfg)
        for r in JsonlLog(p, id_field="index").rows():
            facts_of[f"babilong-{name}-256k-{int(r['index']):03d}"] = [f["text"] for f in r["facts"]]
    sim_check = Counter()

    def cause(r, kv_row_of_text=None):
        sel = r.get("selection") or {}
        kept = sel.get("kept_pages")
        ranks = sel.get("evidence_page_ranks") or []
        task = r["task"]
        if task.startswith("babilong:"):
            name = task.split(":")[1]
            facts = facts_of.get(r["example_id"])
            if name in ("qa1", "qa2") and facts is not None:
                sup = supporting_facts(name, facts, r["question"], r["answers"][0])
                sim_check[name, sup is not None] += 1
                if sup is not None and len(ranks) == len(facts) and kept:
                    miss = [i for i in sup if ranks[i] > kept]
                    det = {"supporting_facts": [facts[i] for i in sup], "supporting_page_ranks": [ranks[i] for i in sup], "kept_pages": kept}
                    if miss:
                        return "missed page", det
                    if name == "qa1":
                        who = r["question"].strip().rstrip("?").split()[-1]
                        earlier = {m.group(2) for f in facts[:sup[0]] for m in [MOVE.match(f.strip())] if m and m.group(1) == who}
                        pred = (r.get("prediction") or "").lower()
                        if any(re.search(rf"\b{re.escape(x)}\b", pred) for x in earlier - {r['answers'][0]}):
                            return "reader error (stale fact)", det
                    return "reader error", det
            rec = r.get("evidence_recall")
            if rec is not None and rec < 1.0:
                return "facts missing (cause undetermined)", {"evidence_recall": rec}
            return "reader error", {"evidence_recall": rec}
        rec = r.get("evidence_recall")
        if rec is None:
            return "reader error (no evidence spans)", {}
        if rec < 1.0:
            return "missed page", {"evidence_recall": rec, "evidence_page_ranks": ranks, "kept_pages": kept}
        return "reader error", {"evidence_recall": rec, "evidence_page_ranks": ranks, "kept_pages": kept}

    fail_rows = []
    by_cause = defaultdict(list)
    for t in retained + infb:
        L = "full" if t.startswith("infbench:") else "256k"
        for i, r in sorted(cell("P-T", t, L).items()):
            if r["score"] < 1.0:
                c, det = cause(r)
                fam = next((f for f, ts in fams.items() if t in ts), "InfiniteBench")
                by_cause[(fam, c)].append({"id": i, "task": t, "length": L, "question": r.get("question"), "answers": r.get("answers"),
                                           "prediction": r.get("prediction"), "cause": c, "detail": det})
        kv = cell("P-KV", t, L)
        tx = cell("P-T", t, L)
        for i in sorted(set(kv) & set(tx)):
            a, b = kv[i], tx[i]
            if a["score"] < 1.0 and b["score"] == 1.0:
                fam = next((f for f, ts in fams.items() if t in ts), "InfiniteBench")
                same_units = (a.get("selection") or {}).get("units_b64") == (b.get("selection") or {}).get("units_b64")
                by_cause[(fam, "stitching")].append({"id": i, "task": t, "length": L, "question": a.get("question"), "answers": a.get("answers"),
                                                     "prediction": a.get("prediction"), "prediction_text_mode": b.get("prediction"),
                                                     "cause": "stitching", "detail": {"same_pages_as_text": same_units,
                                                                                      "stitched_boundaries": (a.get("selection") or {}).get("stitched_boundaries")}})
    by_task = defaultdict(Counter)
    for (fam, c), v in by_cause.items():
        for e in v:
            by_task[e["task"]][c] += 1
    by_task = {t: dict(c) for t, c in sorted(by_task.items())}
    # S2 multi-key at 256K: wrong answers that name one of the three look-alike needles' numbers
    from pcm.tasks.synth import data_file

    distract = {}
    for r_ in JsonlLog(data_file("S2_multikey", "256k", 0, tcfg)).rows():
        distract[r_["id"]] = [re.findall(r"(\d+)\.?$", d["text"])[0] for d in r_["distractors"]]
    s2_distractor = {}
    for k_ in ("P-T", "P-KV"):
        wrong = [r for r in cell(k_, "synth:S2_multikey", "256k").values() if r["score"] < 1.0]
        s2_distractor[k_] = {"wrong": len(wrong), "names_a_distractor_value": sum(1 for r in wrong if any(v in (r.get("prediction") or "") for v in distract.get(r["example_id"], [])))}
    counts = {f"{fam}|{c}": len(v) for (fam, c), v in sorted(by_cause.items(), key=lambda kv: -len(kv[1]))}
    # Ten examples by a fixed rule: groups ordered by size, one example per group in turn (the first by id).
    groups = sorted(by_cause.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ten, k = [], 0
    while len(ten) < 10 and any(len(v) > k for _, v in groups):
        for (fam, c), v in groups:
            if len(v) > k and len(ten) < 10:
                ten.append({"family": fam, **v[k]})
        k += 1
    pt_fail_total = sum(len(v) for (f, c), v in by_cause.items() if c != "stitching")

    # ---- reuse and integrity checks
    checks = {}
    e8 = exp_rows("E8_local")
    checks["e8_status"] = dict(Counter(r["status"] for r in e8))
    checks["e8_commits"] = dict(Counter(f"{(r.get('git_commit') or '')[:10]}{'+dirty' if r.get('git_dirty') else ''}" for r in e8))
    first100 = [r for r in e8 if r.get("method_hash") in (h_pt, h_m2) and r.get("index") is not None and r["index"] < 100 and r["status"] == "ok"]
    checks["first100_reused"] = {"rows": len(first100), "reused": sum(1 for r in first100 if (r.get("timings") or {}).get("reused"))}
    above = []
    for fam in fams:
        for L in LONG:
            a, b = curves.get(f"P-T|{fam}", {}).get(L), curves.get(f"M3-tight|{fam}", {}).get(L)
            if a and b and a["lo"] > b["hi"]:
                above.append(f"P-T above the tight oracle: {fam} {L}")
            if a and ref[fam]["mean"] is not None and a["lo"] > ref[fam]["hi"]:
                above.append(f"P-T above the native 8K reference: {fam} {L}")
    checks["too_good"] = above
    checks["supporting_fact_replay"] = {f"{n}|{ok}": v for (n, ok), v in sim_check.items()}
    checks["status_by_series"] = {k: dict(v) for k, v in all_status.items()}

    # ---- what remains for the rented machine: GPU-hours from measured seconds per example (projections)
    def sec(k, L, tasks):
        rs = [r for t in tasks for r in cell(k, t, L).values()]
        fr = fresh(rs)
        return mean([(r.get("timings") or {}).get("total_s", 0) - (r.get("timings") or {}).get("ingest_s", 0.0) for r in fr])

    synth = cfg["task_sets"]["synth"]
    m1 = {L: sec("M1", L, retained) for L in ("64k", "128k")}
    growth = m1["128k"] / m1["64k"] if m1["64k"] and m1["128k"] else None
    proj = {"m1_seconds": {L: rnd(v, 2) for L, v in m1.items()}, "m1_growth_per_doubling": rnd(growth, 3)}
    if growth:
        s256 = m1["128k"] * growth
        proj["m1_256k_seconds_extrapolated"] = rnd(s256, 1)
        proj["m1_256k_hours"] = {str(n): rnd(s256 * n * 13 / 3600, 1) for n in (30, 50, 100)}
    grid_h = {}
    for k in ("M0", "M2", "M3", "P-T", "P-KV"):
        h = 0.0
        for L in LONG:
            s_ = sec(k, L, synth)
            n = 300 if (L == "256k" and k in ("M2", "P-T", "P-KV")) else 100
            if s_:
                h += s_ * n * len(synth) * 2 / 3600     # seeds 1 and 2 on the synthetic tasks (BABILong is one fixed test set)
        grid_h[k] = rnd(h, 2)
    m1_seed = sum((m1[L] or 0) * 40 * len(synth) * 2 for L in m1) / 3600
    grid_h["M1 (64K, 128K, N = 40)"] = rnd(m1_seed, 2)
    ing = t256["ingest_log_256k"]["embed_s"] or 0
    grid_h["ingest of new synthetic documents"] = rnd(2 * len(synth) * (100 * (1.06 + 1.92 + 3.84) / 3 + 300 * ing) / 3600, 2)
    proj["final_grid_two_more_seeds_hours"] = grid_h
    proj["final_grid_total_hours"] = rnd(sum(v for v in grid_h.values() if v), 2)
    pt_q = sec("P-T", "256k", retained)
    tps = t256["ingest_log_256k"]["tokens_per_s"]
    if tps and pt_q:
        proj["e12_pcm_text"] = {"seconds_per_question_256k": rnd(pt_q, 3), "embed_tokens_per_s": tps,
                                "hours_512k_1m_13_tasks_n100": rnd((pt_q * 2 + (524288 + 1048576) / tps) * 13 * 100 / 3600, 2)}

    # ---- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlecolor": "#0b0b0b", "font.size": 9})
    native_curve = {}
    for fam, tasks in fams.items():
        native_curve[fam] = {L: family_ci([list(scores("native", t, L).values()) for t in tasks], mc) for L in ("8k", "16k", "32k")}
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, fam in zip(axes, fams):
        nc = native_curve[fam]
        nL = [L for L in ("8k", "16k", "32k") if nc[L]["mean"] is not None]
        ax.plot([GRID.index(L) for L in nL], [100 * nc[L]["mean"] for L in nL], "-", color="#8a8984", marker="o", markersize=5,
                linewidth=2.2, label="native (whole document in the window)", zorder=1)
        for k in ["M0", "M1", "M2", "M3", "M3-tight", "P-KV", "P-T"]:
            cur = curves.get(f"{k}|{fam}", {})
            own = [L for L in GRID if L in cur and cur[L]["source"] == "method"]
            if not own:
                continue
            col, mk, ls = STYLE[k]
            x = [GRID.index(L) for L in own]
            y = [100 * cur[L]["mean"] for L in own]
            ax.fill_between(x, [100 * cur[L]["lo"] for L in own], [100 * cur[L]["hi"] for L in own], color=col, alpha=0.12, linewidth=0)
            first = GRID.index(own[0])
            if first > 0 and GRID[first - 1] in cur:          # leave the native curve at the length before the method's own rows
                ax.plot([first - 1, first], [100 * cur[GRID[first - 1]]["mean"], y[0]], ls, color=col, linewidth=1.0, alpha=0.7)
            ax.plot(x, y, ls, color=col, marker=mk, markersize=6, linewidth=2.2 if k in ("P-T", "P-KV") else 1.4, label=ser[k]["label"],
                    markeredgecolor="#fcfcfb", markeredgewidth=1.0, zorder=3 if k == "P-T" else 2)
        if ref[fam]["mean"] is not None:
            thr = 100 * mc["ecl_ratio"] * ref[fam]["mean"]
            ax.axhline(thr, color="#52514e", linewidth=0.9, linestyle="-.", label="ECL90 threshold: 90% of native 8K")
        ax.set_title(fam + ("" if fam in h1fams else " (no ECL90, D-032)"), fontsize=10)
        ax.set_xticks(range(len(GRID)))
        ax.set_xticklabels([L.upper() for L in GRID], fontsize=8)
        ax.set_ylim(-2, 102)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.set_xlabel("document length (tokens)")
    axes[0].set_ylabel("accuracy (%), 95% CI band")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("E8: accuracy versus length, Qwen3-1.7B with a 32K window (one data seed)", fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "curves.png", dpi=130)
    plt.close(fig)
    figs = ["figures/curves.png"]
    # 256K: accuracy per family per method
    keys = [k for k in ("M0", "M2", "M3", "P-KV", "P-T") if k in t256]
    if keys:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        w = 0.8 / len(keys)
        for j, k in enumerate(keys):
            col, _, _ = STYLE[k]
            ys = [100 * (t256[k]["accuracy"][f]["mean"] or 0) for f in fams]
            xs = [i + (j - (len(keys) - 1) / 2) * w for i in range(len(fams))]
            ax.bar(xs, ys, w * 0.92, color=col, label=ser[k]["label"])
        for i, f in enumerate(fams):
            if ref[f]["mean"] is not None:
                ax.hlines(100 * ref[f]["mean"], i - 0.42, i + 0.42, color="#0b0b0b", linewidth=1.2)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(list(fams))
        ax.set_ylabel("accuracy at 256K (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
        ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper right")
        ax.set_title("256K: accuracy by method; black line = native accuracy at 8K", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "acc_256k.png", dpi=130)
        plt.close(fig)
        figs.append("figures/acc_256k.png")

    summary = {"commits": {k: dict(v) for k, v in commits.items()}, "reference": ref, "curves": curves, "ecl90": ecl, "ecl90_alternative": alt,
               "pairs": pairs, "kv_gap_pooled": kv_gap, "codec_int4_vs_bf16": codec, "infbench": inf, "table_256k": t256, "store": store,
               "gate": gate, "h5": h5, "failures": {"counts": counts, "ten": ten, "pt_failures_256k": pt_fail_total, "s2_multikey_distractor": s2_distractor, "by_task": by_task}, "checks": checks,
               "projection": proj, "figures": figs, "hashes": {"P-T": h_pt, "M2": h_m2, "P-KV": h_kv}}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "summary.json", summary)

    # ---- tables.md
    md = ["# A8 tables (generated by scripts/a8_summary.py)", ""]
    md += ["## Accuracy by family and length (%, 95% CI; n per task in brackets)", ""]
    for fam in fams:
        md += [f"### {fam} (native 8K reference {pct(ref[fam])})", "", "| series | " + " | ".join(GRID) + " |", "|---|" + "---|" * len(GRID)]
        for k in ["M0", "M1", "M2", "M3", "M3-tight", "P-T", "P-KV"]:
            cur = curves.get(f"{k}|{fam}", {})
            cells_ = []
            for L in GRID:
                c = cur.get(L)
                cells_.append("–" if not c else (p0(c) + " (native)" if c["source"] == "native" else f"{pct(c)} (n {min(c['n_per_task'])})"))
            md.append(f"| {ser[k]['label']} | " + " | ".join(cells_) + " |")
        md.append("")
    md += ["## ECL90", "", "| series | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for k in ["M0", "M1", "M2", "M3", "M3-tight", "P-T", "P-KV"]:
        md.append(f"| {ser[k]['label']} | " + " | ".join(str((ecl.get(f'{k}|{f}') or {}).get('ecl')) for f in fams) + " |")
    md += ["", "Alternative readings (D-053 (4)): " + json.dumps(alt), ""]
    md += ["## Paired differences (points, 95% CI)", "", "| pair | family | " + " | ".join(LONG) + " |", "|---|---|" + "---|" * len(LONG)]
    for a, b in pair_list:
        for fam in fams:
            md.append(f"| {a} − {b} | {fam} | " + " | ".join(dpct(pairs.get(f"{a}-{b}|{fam}|{L}")) for L in LONG) + " |")
    md += ["", "## KV − text pooled over 13 tasks; int4 − bf16 store (KV mode)", "", "| length | KV | text | KV − text | int4 − bf16 | same answer |", "|---|---|---|---|---|---|"]
    for L in LONG:
        g_, c_ = kv_gap.get(L) or {}, codec.get(L) or {}
        md.append(f"| {L} | {f2(100 * g_['kv'] if g_.get('kv') is not None else None, 1)} | {f2(100 * g_['text'] if g_.get('text') is not None else None, 1)} | "
                  f"{dpct(g_)} | {dpct(c_)} | {c_.get('same_answer')} |")
    md += ["", "First 100 examples per task (as A6): " + "; ".join(f"{L}: {dpct((kv_gap.get(L) or {}).get('first100'))}, pairs {((kv_gap.get(L) or {}).get('first100') or {}).get('pairs')}" for L in LONG)]
    md += ["", "## InfiniteBench", "", "| series | " + " | ".join(infb) + " |", "|---|" + "---|" * len(infb)]
    for k in S:
        md.append(f"| {ser[k]['label']} | " + " | ".join(pct((inf.get(f'{k}|{t}') or {}).get('accuracy')) for t in infb) + " |")
    md += ["", "## 256K", "", "```", json.dumps(t256, indent=1), "```", "", "## Store check", "", "```", json.dumps({k: v for k, v in store.items() if k != "rows"}, indent=1), "```"]
    md += ["", "## Gate G3", "", "```", json.dumps(gate, indent=1), "```", "", "## H5 (A3 rows)", ""]
    md += [f"- {k}: {v['method']}, n {v['n']}, {pct(v['accuracy'])}" for k, v in h5.items()]
    md += ["", "## Failure causes at 256K (P-T; stitching = P-KV wrong where P-T right)", ""] + [f"- {k}: {v}" for k, v in counts.items()]
    md += ["", "By task: " + json.dumps(by_task)]
    md += ["", f"S2 multi-key at 256K, wrong answers naming a look-alike needle's number: {json.dumps(s2_distractor)}"]
    md += ["", "## Checks", "", "```", json.dumps(checks, indent=1), "```", "", "## Projection", "", "```", json.dumps(proj, indent=1), "```"]
    atomic_write_text(OUT / "tables.md", "\n".join(md) + "\n")
    print(json.dumps({"gate": gate, "kv_gap": {L: dpct(v) for L, v in kv_gap.items()}, "codec": {L: dpct(v) for L, v in codec.items()},
                      "checks": checks}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
