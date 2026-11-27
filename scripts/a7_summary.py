"""Numbers, tables and figures for reports/A7_codecs.md.

    python scripts/a7_summary.py

Reads results/E7_codecs (part1.jsonl, part2.jsonl, part4.jsonl, basis.json, pages.json, coder_choice.json,
lowrank_choice.json and, if Part 3 ran, runs.jsonl) and, for Part 3's bf16 reference, results/E6_kv. Decisions D-049
and D-050. Writes results/A7_codecs/summary.json, tables.md and figures/*.png. Every table gives n.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pcm.eval import metrics as M  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_json, atomic_write_text  # noqa: E402
from pcm.tasks.common import config_hash, load_task_config  # noqa: E402

E7 = REPO_ROOT / "results" / "E7_codecs"
OUT = REPO_ROOT / "results" / "A7_codecs"
FIG = OUT / "figures"
BF16_PER_TOKEN = 114688
GIB = 1024**3


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    last = {}
    for r in JsonlLog(path).rows():
        last[r["id"]] = r
    return list(last.values())


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def f(x, k=2):
    return "–" if x is None else f"{x:.{k}f}"


def dpct(d, digits=1):
    if not d or d.get("diff") is None:
        return "–"
    return f"{100 * d['diff']:+.{digits}f} [{100 * d['lo']:+.{digits}f}, {100 * d['hi']:+.{digits}f}]"


def main() -> int:
    tcfg = load_task_config()
    mc = tcfg["metrics"]
    B, SEED, CI = mc["bootstrap_resamples"], mc["bootstrap_seed"], mc["ci"]
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "a7_codecs.yaml").read_text(encoding="utf-8"))
    p1, p2, p4 = rows(E7 / "part1.jsonl"), rows(E7 / "part2.jsonl"), rows(E7 / "part4.jsonl")
    basis, pages = load(E7 / "basis.json"), load(E7 / "pages.json")
    coder_choice, lowrank_choice = load(E7 / "coder_choice.json"), load(E7 / "lowrank_choice.json")
    commits = {n: dict(Counter((r.get("git_commit") or "")[:10] + ("+dirty" if r.get("git_dirty") else "") for r in rs))
               for n, rs in (("part1", p1), ("part2", p2), ("part4", p4))}

    # ---- Part 1
    by = defaultdict(list)
    for r in p1:
        by[r["codec"]].append(r)
    timing = defaultdict(list)
    for r in rows(E7 / "timing.jsonl"):
        timing[r["codec"]].append(r)
    commits["timing"] = dict(Counter((r.get("git_commit") or "")[:10] for rs in timing.values() for r in rs))
    part1 = {}
    for c, rs in by.items():
        tr = timing.get(c, [])
        part1[c] = {"pages": len(rs), "bytes_per_token": mean([r["bytes_per_token"] for r in rs]), "ratio": mean([r["ratio"] for r in rs]),
                    "shared_bytes": rs[0].get("shared_bytes", 0), "timing_pages": len(tr),
                    "encode_ms": float(np.median([r["encode_ms"] for r in tr])) if tr else None,
                    "decode_ms": float(np.median([r["decode_ms"] for r in tr])) if tr else None,
                    "rel_err_k": mean([r["rel_err_k"] for r in rs]), "rel_err_v": mean([r["rel_err_v"] for r in rs]),
                    "kl": M.bootstrap_ci([r["kl"] for r in rs], B, SEED, CI), "kl_max": max(r["kl_max"] for r in rs),
                    "top1": M.bootstrap_ci([r["top1"] for r in rs], B, SEED, CI)}

    # ---- Part 2
    agg2 = defaultdict(lambda: {"raw": 0, "comp": 0, "cs": 0.0, "ds": 0.0, "n": 0})
    for r in p2:
        a = agg2[(r["item"], r["coder"])]
        a["raw"] += r["raw_bytes"]
        a["comp"] += r["compressed_bytes"]
        a["cs"] += r["compress_s"]
        a["ds"] += r["decompress_s"]
        a["n"] += 1
    part2 = {f"{i}|{c}": {"n": a["n"], "raw_mb": a["raw"] / 1e6, "ratio": a["raw"] / a["comp"], "compress_mb_s": a["raw"] / 1e6 / a["cs"],
                          "decompress_mb_s": a["raw"] / 1e6 / a["ds"]} for (i, c), a in agg2.items()}
    # bytes per token after the coder, for the KV items (per page of 1,024 tokens)
    for k, v in part2.items():
        item = k.split("|")[0]
        if item in ("bf16", "bf16-shuffle", "int8", "int4", "int2"):
            per_tok = (v["raw_mb"] * 1e6 / v["n"]) / cfg["pages"]["heldout"]["page_tokens"]
            v["bytes_per_token_after"] = per_tok / v["ratio"]
            v["ratio_vs_bf16"] = BF16_PER_TOKEN / v["bytes_per_token_after"]

    # ---- Part 3
    part3 = {}
    p3 = rows(E7 / "runs.jsonl")
    if p3:
        import run_a6

        cfg6 = run_a6.load_config()
        ape = run_a6.ape_choice(cfg6)
        ref_m = run_a6.mode_config(cfg6, "kv-ape", ape=ape)
        base_name = ref_m["name"]
        e6 = rows(REPO_ROOT / "results" / "E6_kv" / "runs.jsonl")
        ref = defaultdict(dict)
        for r in e6:
            if r["status"] == "ok" and r["method_hash"] == config_hash(ref_m):
                ref[(r["task"], r["length"])][r["example_id"]] = r["score"]
        got = defaultdict(dict)
        for r in p3:
            if r["status"] == "ok":
                got[(r["method"], r["task"], r["length"])][r["example_id"]] = r
        for m in sorted({m for (m, _, _) in got}):
            codec = m[len(base_name) + 1:]
            for L in cfg["part3"]["lengths"]:
                a, b, per_task, exp_ms = [], [], {}, []
                for t in cfg["part3"]["tasks"]:
                    g, rf = got.get((m, t, L), {}), ref.get((t, L), {})
                    ids = sorted(set(g) & set(rf))
                    x, y = [g[i]["score"] for i in ids], [rf[i] for i in ids]
                    if ids:
                        per_task[t] = {**M.paired_bootstrap(x, y, B, SEED, CI), "codec_mean": mean(x), "bf16_mean": mean(y)}
                    a += x
                    b += y
                    exp_ms += [1000 * g[i]["timings"]["expand_s"] for i in ids if g[i]["timings"].get("expand_s") is not None]
                if a:
                    part3[f"{codec}|{L}"] = {**M.paired_bootstrap(a, b, B, SEED, CI), "codec_mean": mean(a), "bf16_mean": mean(b), "tasks": per_task,
                                             "expand_ms": mean(exp_ms)}

    # ---- Part 4
    sizes = next((r for r in p4 if r["id"] == "sizes"), None)
    lat = [r for r in p4 if r["id"].startswith("latency|")]
    latency = {}
    if lat:
        for c in lat[0]["codecs"]:
            latency[c] = float(np.median([r["codecs"][c]["expand_ms"] for r in lat]))
        latency["text-mode prefill"] = float(np.median([r["recompute_text_mode_ms"] for r in lat]))
    vram = {r["codec"]: r for r in p4 if r["id"].startswith("vram|")}
    env = load(E7 / "env.json") or {}
    device_gib = float(((env.get("gpus") or [{}])[0]).get("vram_total_gib") or cfg["part4"]["vram_gib"])
    for c, r in vram.items():
        # Fits = it ran without an out-of-memory error AND the allocator's peak reserved memory stayed within the card;
        # beyond it, the Windows driver spills into system memory (A4) and the run completes without fitting.
        r["fits_on_card"] = bool(r.get("attempted") and r.get("fits") and (r.get("peak_reserved_gib") or 1e9) <= device_gib)

    # ---- H4
    h4 = {"lossless_alone_on_bf16": None, "candidates": []}
    raw_best = max((v["ratio"], k) for k, v in part2.items() if k.startswith("bf16|")) if any(k.startswith("bf16|") for k in part2) else None
    shuf_best = max((v["ratio"], k) for k, v in part2.items() if k.startswith("bf16-shuffle|")) if any(k.startswith("bf16-shuffle|") for k in part2) else None
    h4["lossless_alone_on_bf16"] = {"raw": raw_best, "shuffled": shuf_best}
    for c, v in part1.items():
        if v["ratio"] >= 4:
            entry = {"codec": c, "ratio": v["ratio"], "kl": v["kl"]["mean"], "top1": v["top1"]["mean"]}
            losses = {L: part3.get(f"{c[:-5] if c.startswith('int') and c.endswith('-kcvt') else c}|{L}") for L in cfg["part3"]["lengths"]}
            entry["part3"] = {L: (None if d is None else d["diff"]) for L, d in losses.items()}
            entry["holds"] = all(d is not None and d["diff"] >= -0.01 - 1e-9 for d in losses.values()) if part3 else None
            h4["candidates"].append(entry)

    # ---- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    figs = []
    if part1:
        fig, axes = plt.subplots(1, 2 if part3 else 1, figsize=(13 if part3 else 7, 4.4))
        axes = np.atleast_1d(axes)
        ax = axes[0]
        for c, v in part1.items():
            if c == "text":
                continue
            color = "tab:blue" if c.startswith("int8") else "tab:orange" if c.startswith("int4") else "tab:red" if c.startswith("int2") else "tab:green" if c.startswith("lowrank") else "black"
            marker = "o" if c.endswith("kcvt") or not c.startswith("int") else "x"
            ax.scatter(v["bytes_per_token"], max(v["kl"]["mean"], 1e-6), color=color, marker=marker, s=28)
            ax.annotate(c, (v["bytes_per_token"], max(v["kl"]["mean"], 1e-6)), fontsize=6, xytext=(3, 2), textcoords="offset points")
        # A lossless coder leaves the decoded cache, hence the KL, unchanged: int8/int4/int2 (KIVI) + the best coder of Part 2.
        for q in ("int8", "int4", "int2"):
            best = max(((v["ratio"], k) for k, v in part2.items() if k.startswith(q + "|")), default=None)
            if best and f"{q}-kcvt" in part1:
                bpt = part2[best[1]]["bytes_per_token_after"]
                kl = part1[f"{q}-kcvt"]["kl"]["mean"]
                ax.scatter(bpt, kl, color="purple", marker="^", s=30)
                ax.annotate(f"{q} + {best[1].split('|')[1]}", (bpt, kl), fontsize=6, xytext=(3, -8), textcoords="offset points")
        ax.axvline(BF16_PER_TOKEN / 4, color="black", linestyle=":", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("stored bytes per token (log; dotted: 4x smaller than bf16)")
        ax.set_ylabel("next-token KL vs bf16 (log)")
        ax.set_title("Accuracy loss stand-in (Part 3 not run: Gate G2 failed): next-token KL on 256 held-out PG-19 pages (bf16: KL 0, drawn at 1e-6)", fontsize=8)
        if part3:
            ax = axes[1]
            for k, d in part3.items():
                c, L = k.split("|")
                bpt = part1.get(f"{c}-kcvt" if c in ("int8", "int4", "int2") else c, {}).get("bytes_per_token")
                if bpt is None:
                    continue
                ax.errorbar(bpt, -100 * d["diff"], yerr=[[100 * (d["hi"] - d["diff"])], [100 * (d["diff"] - d["lo"])]], fmt="o" if L == "256k" else "s",
                            color="tab:green" if L == "256k" else "tab:blue", capsize=2)
                ax.annotate(f"{c} {L}", (bpt, -100 * d["diff"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
            ax.axhline(1, color="tab:red", linestyle=":", linewidth=0.8)
            ax.axvline(BF16_PER_TOKEN / 4, color="black", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xlabel("stored bytes per token (log)")
            ax.set_ylabel("accuracy loss vs bf16 (points, 95% CI)")
            ax.set_title("Part 3: KV mode, 8 tasks pooled; red dotted: 1 point (H4)", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "pareto.png", dpi=120)
        plt.close(fig)
        figs.append("figures/pareto.png")

    summary = {"device_gib": device_gib, "commits": commits, "basis": basis, "pages": {k: len(v) for k, v in (pages or {}).items() if isinstance(v, list)},
               "part1": part1, "part2": part2, "part3": part3, "part4": {"sizes": sizes, "latency_ms": latency, "vram": vram},
               "coder_choice": coder_choice, "lowrank_choice": lowrank_choice, "h4": h4, "figures": figs}
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(OUT / "summary.json", summary)

    out = ["# A7 tables (generated by scripts/a7_summary.py from results/E7_codecs)", "", f"Rows by commit: {commits}", ""]
    if basis:
        out += [f"Basis: {basis['tokens']} calibration tokens ({basis['pages']} pages); explained variance, mean over layers (min): "
                + ", ".join(f"{k} {v['mean']:.3f} ({v['min']:.3f})" for k, v in basis["explained"].items()), ""]
    out += ["## Part 1: storage and fidelity, 256 held-out pages of 1,024 tokens (means; KL and top-1 with 95% CI over pages; encode and decode ms: median over the 32 pages of the timing pass, each a median of 5)", "",
            "| codec | bytes/token | ratio vs bf16 | encode ms | decode ms | rel. error K | rel. error V | next-token KL | top-1 agreement | max KL |", "|---" * 10 + "|"]
    for c, v in sorted(part1.items(), key=lambda kv: -kv[1]["bytes_per_token"]):
        out.append(f"| {c} | {v['bytes_per_token']:.0f} | {v['ratio']:.2f} | {f(v['encode_ms'], 1)} | {f(v['decode_ms'], 1)} | {v['rel_err_k']:.4f} | {v['rel_err_v']:.4f} | "
                   f"{v['kl']['mean']:.5f} [{v['kl']['lo']:.5f}, {v['kl']['hi']:.5f}] | {v['top1']['mean']:.4f} [{v['top1']['lo']:.4f}, {v['top1']['hi']:.4f}] | {v['kl_max']:.3f} |")
    shared = {c: v["shared_bytes"] for c, v in part1.items() if v["shared_bytes"]}
    if shared:
        out += ["", "Shared bytes (bases, once per store): " + ", ".join(f"{c} {b / 1e6:.1f} MB" for c, b in shared.items())]
    if part2:
        items = sorted({k.split("|")[0] for k in part2}, key=lambda s: ["bf16", "bf16-shuffle", "int8", "int4", "int2", "ids-uint32", "text-utf8"].index(s) if s in ["bf16", "bf16-shuffle", "int8", "int4", "int2", "ids-uint32", "text-utf8"] else 99)
        coders = cfg["part2"]["coders"]
        out += ["", "## Part 2: lossless coders, ratio (compress / decompress MB/s)", "", "| data | MB | " + " | ".join(coders) + " |", "|---" * (2 + len(coders)) + "|"]
        for it in items:
            first = next(v for k, v in part2.items() if k.startswith(it + "|"))
            out.append(f"| {it} | {first['raw_mb']:.1f} | " + " | ".join(
                f"{part2[f'{it}|{c}']['ratio']:.3f} ({part2[f'{it}|{c}']['compress_mb_s']:.0f} / {part2[f'{it}|{c}']['decompress_mb_s']:.0f})" if f"{it}|{c}" in part2 else "–" for c in coders) + " |")
        out += ["", "Bytes per token after the coder, and ratio against bf16:", "", "| data | " + " | ".join(coders) + " |", "|---" * (1 + len(coders)) + "|"]
        for it in ("bf16", "bf16-shuffle", "int8", "int4", "int2"):
            if any(k.startswith(it + "|") for k in part2):
                out.append(f"| {it} | " + " | ".join(f"{part2[f'{it}|{c}']['bytes_per_token_after']:.0f} ({part2[f'{it}|{c}']['ratio_vs_bf16']:.2f}x)" if f"{it}|{c}" in part2 else "–" for c in coders) + " |")
        if coder_choice:
            out += ["", f"Coder for int4 + lossless: {coder_choice['choice']} ({coder_choice['rule']})"]
    if lowrank_choice:
        out += ["", f"Best low-rank setting: {lowrank_choice['choice']} ({lowrank_choice['rule']}; int4 KL {lowrank_choice['int4_kl']:.5f}; qualified {lowrank_choice['qualified']})"]
    if part3:
        out += ["", "## Part 3: KV mode accuracy with each codec, paired against bf16 (E6 mode (c)), pooled over the 8 tasks, points [95% CI]", "",
                "| codec | length | n | bf16 | codec | loss (codec - bf16) | expansion ms |", "|---|---|---|---|---|---|---|"]
        for k, d in part3.items():
            c, L = k.split("|")
            out.append(f"| {c} | {L} | {d['n']} | {100 * d['bf16_mean']:.1f} | {100 * d['codec_mean']:.1f} | {dpct(d)} | {d['expand_ms']:.0f} |")
    if sizes:
        out += ["", f"## Part 4: whole store of a 256K document ({sizes['example']}, {sizes['tokens']} tokens, {sizes['pages']} pages of {sizes['page']})", "",
                "| codec | store GiB | shared MB | bytes/token | store + weights + 32K window (arithmetic) | measured: on the card / peak allocated / peak reserved GiB |", "|---|---|---|---|---|---|"]
        for c, v in sizes["codecs"].items():
            vr = vram.get(c)
            arith = None if vr is None else f"{vr['arithmetic_gib']:.1f} GiB of {vr['vram_gib']:.0f}"
            meas = "–" if vr is None else ("not attempted" if not vr.get("attempted") else
                                           f"{'yes' if vr['fits_on_card'] else 'no'} / {f(vr.get('peak_allocated_gib'), 1)} / {f(vr.get('peak_reserved_gib'), 1)}")
            out.append(f"| {c} | {v['store_bytes'] / GIB:.2f} | {v['shared_bytes'] / 1e6:.1f} | {v['bytes_per_token']:.0f} | {arith or '–'} | {meas} |")
        out += ["", f"Model weights: {sizes['weights_bytes'] / GIB:.2f} GiB. Active window of 32,768 tokens in bf16: {32768 * BF16_PER_TOKEN / GIB:.2f} GiB. "
                f"Card: {device_gib:.1f} GiB. On the card = no out-of-memory error and peak reserved memory within the card. "
                f"Ingest of the whole document with every codec: {sizes['ingest_s_all_codecs']:.0f} s."]
    if latency:
        out += ["", "Expansion of a full budget (56 pages, 28,672 tokens), median of 3 (ms):", "", "| codec | ms |", "|---|---|"]
        for c, v in latency.items():
            out.append(f"| {c} | {v:.0f} |")
    out += ["", f"## H4: {json.dumps(h4, default=str)}"]
    atomic_write_text(OUT / "tables.md", "\n".join(out) + "\n")
    print(f"wrote {OUT / 'summary.json'} and tables.md; figures {figs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
