"""Numbers, figures and examples for reports/A2_tasks.md, from the files under results/A2_tasks/.

    python scripts/a2_summary.py

Reads synth_manifest.jsonl, lengths_babilong.jsonl, lengths_infbench.jsonl and
sizes.json (written by scripts/make_data.py), and one example per task from data/.
Writes results/A2_tasks/summary.json, examples.json and figures/*.png, and prints
the tables as Markdown.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.models import chat_segments, yarn_factor_for  # noqa: E402
from pcm.runlog import atomic_write_json  # noqa: E402
from pcm.tasks.common import get_tokenizer, iter_jsonl, load_task_config  # noqa: E402

OUT = REPO_ROOT / "results" / "A2_tasks"
FIG = OUT / "figures"
WINDOW = 32768
ORIGINAL_POSITIONS = 40960
EXCERPT = 160


def yarn_factor(max_tokens: int):
    """Smallest YaRN factor in (2, 4, 8, 16) covering max_tokens; None if no extension is needed; "over 16" if none suffices."""
    try:
        return yarn_factor_for(max_tokens, ORIGINAL_POSITIONS)
    except ValueError:
        return "over 16"


def q(xs: list[int]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "min": xs[0], "median": statistics.median(xs), "mean": round(statistics.fmean(xs), 1), "max": xs[-1]}


def nominal_tokens(name: str) -> int | None:
    if name.endswith("k"):
        return int(name[:-1]) * 1024
    if name.endswith("M"):
        return int(name[:-1]) * 1024 * 1024
    return None


def synth_summary(cfg: dict) -> dict:
    rows = list(iter_jsonl(OUT / "synth_manifest.jsonl"))
    tol = cfg["synth"]["length_tolerance"]
    cells: dict = defaultdict(list)
    for r in rows:
        cells[(r["task"], r["length_name"], r["seed"])].append(r)
    out = {}
    for (task, L, seed), rs in sorted(cells.items(), key=lambda k: (k[0][0], cfg["lengths"].get(k[0][1], 0), k[0][2])):
        target = rs[0]["length_nominal"]
        lens = [r["length_tokens"] for r in rs]
        dec = Counter(r["depth_decile"] for r in rs if r["depth_decile"] is not None)
        n_ev = [len(r["evidence"]) for r in rs]
        out[f"{task}|{L}|{seed}"] = {
            "task": task, "length": L, "seed": seed, "target": target, "tokens": q(lens),
            "within_tolerance": sum((1 - tol) * target <= x <= target for x in lens),
            "max_rel_shortfall": round(max((target - x) / target for x in lens), 5),
            "deciles": dict(sorted(dec.items())), "evidence_items": q(n_ev),
            "attempts_mean": round(statistics.fmean(r["attempts"] for r in rs), 3),
            "first_books_distinct": len({r["books"][0] for r in rs}),
            "books_per_example": q([len(r["books"]) for r in rs]),
        }
    return out


def babilong_summary() -> tuple[dict, dict, list[dict]]:
    rows = list(iter_jsonl(OUT / "lengths_babilong.jsonl"))
    by_len: dict = defaultdict(list)
    for r in rows:
        by_len[r["length_name"]].append(r)
    lengths = {}
    for L, rs in by_len.items():
        toks = [r["length_tokens"] for r in rs]
        prompts = [r["prompt_tokens"] for r in rs]
        nom = nominal_tokens(L)
        lengths[L] = {
            "nominal": nom, "context_tokens": q(toks), "prompt_tokens": q(prompts),
            "ratio_median_to_nominal": round(statistics.median(toks) / nom, 4) if nom else None,
            "prompts_within_32768": sum(p <= WINDOW for p in prompts),
            "yarn_factor_for_max_prompt": yarn_factor(max(prompts)),
        }
    loc: dict = defaultdict(lambda: Counter())
    for r in rows:
        l = r.get("locator") or {}
        c = loc[r["task"]]
        c["rows"] += 1
        c["located"] += l.get("located", r["n_facts"])
        c["located_known"] += l.get("located_known", 0)
        if "facts_0k" in l:
            c["aligned_rows"] += 1
            c["facts_0k"] += l["facts_0k"]
            c["matched"] += l["matched"]
            c["rows_exact"] += int(l["matched"] == l["facts_0k"] == l["located"])
    locator = {}
    for t, c in loc.items():
        locator[t] = {**dict(c),
                      "share_located_that_are_0k_facts": round(c["located_known"] / c["located"], 5) if c["located"] else None,
                      "recall_vs_aligned_0k_row": round(c["matched"] / c["facts_0k"], 5) if c["facts_0k"] else None}
    return lengths, locator, rows


def infbench_summary() -> tuple[dict, list[dict]]:
    rows = list(iter_jsonl(OUT / "lengths_infbench.jsonl"))
    by_task: dict = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    out = {}
    for t, rs in by_task.items():
        toks = [r["length_tokens"] for r in rs]
        prompts = [r["prompt_tokens"] for r in rs]
        out[t] = {
            "context_tokens": q(toks), "prompt_tokens": q(prompts), "chars": q([r["chars"] for r in rs]),
            "over_262144": sum(p > 262144 for p in prompts), "over_131072": sum(p > 131072 for p in prompts),
            "over_655360": sum(p > 16 * ORIGINAL_POSITIONS for p in prompts),
            "within_32768": sum(p <= WINDOW for p in prompts),
            "yarn_factor_for_max_prompt": yarn_factor(max(prompts)),
            "rows_with_evidence": sum(r["n_evidence"] > 0 for r in rs),
            "evidence_deciles": dict(sorted(Counter(r["evidence_decile"] for r in rs if r["evidence_decile"] is not None).items())),
        }
    return out, rows


def figures(bab_rows: list[dict], inf_rows: list[dict], cfg: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    lengths = [L for L in cfg["babilong"]["lengths"] if L != "0k"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for ax, L in zip(axes.flat, lengths):
        toks = [r["length_tokens"] for r in bab_rows if r["length_name"] == L]
        nom = nominal_tokens(L)
        ax.hist(toks, bins=30, color="#4C72B0")
        ax.axvline(nom, color="#C44E52", linestyle="--", label=f"nominal {nom:,}")
        ax.set_title(f"BABILong {L} (n={len(toks)})", fontsize=9)
        ax.set_xlabel("context tokens (Qwen3)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("BABILong qa1-qa10: measured Qwen3 token count of the context vs nominal length", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "babilong_lengths.png", dpi=110)
    plt.close(fig)
    made.append("figures/babilong_lengths.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for L in lengths:
        toks = [r["length_tokens"] / nominal_tokens(L) for r in bab_rows if r["length_name"] == L]
        ax.scatter([L] * len(toks), toks, s=4, alpha=0.3)
    ax.axhline(1.0, color="#C44E52", linestyle="--")
    ax.set_ylabel("measured / nominal tokens")
    ax.set_title("BABILong: measured Qwen3 context tokens relative to nominal length")
    fig.tight_layout()
    fig.savefig(FIG / "babilong_ratio.png", dpi=110)
    plt.close(fig)
    made.append("figures/babilong_ratio.png")

    tasks = cfg["infbench"]["tasks"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.3))
    for ax, t in zip(axes, tasks):
        toks = [r["length_tokens"] for r in inf_rows if r["task"] == t]
        lo, hi = min(toks), max(toks)
        pad = max(1000, 0.02 * hi)  # constant-length tasks get a visible bar
        ax.hist(toks, bins=[lo - pad + i * (hi - lo + 2 * pad) / 30 for i in range(31)], color="#55A868")
        if lo == hi:
            ax.text(lo, ax.get_ylim()[1] * 0.5, f" all rows: {lo:,}", fontsize=7)
        for x, lab in ((131072, "128K"), (262144, "256K")):
            ax.axvline(x, color="#C44E52", linestyle="--", linewidth=1)
            ax.text(x, ax.get_ylim()[1] * 0.9, lab, fontsize=7, color="#C44E52")
        ax.set_title(f"InfiniteBench {t} (n={len(toks)})", fontsize=9)
        ax.set_xlabel("context tokens (Qwen3)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "infbench_lengths.png", dpi=110)
    plt.close(fig)
    made.append("figures/infbench_lengths.png")
    return made


def _excerpt(ex: dict, ev: dict | None) -> str:
    ctx = ex["context"]
    if ev is None:
        return ctx[:2 * EXCERPT] + " [...]"
    a, b = ev["char_start"], ev["char_end"]
    return "[...] " + ctx[max(0, a - EXCERPT):a] + "<<" + ctx[a:b] + ">>" + ctx[b:b + EXCERPT] + " [...]"


def examples(cfg: dict) -> dict:
    """One example per task (index 3 of seed 0 at 8k for synthetic tasks; row 0 at 8k / first row otherwise)."""
    from pcm.eval.runner import load_examples, prompt_for
    from pcm.tasks.synth import SYNTH_TASKS, data_file

    tok = get_tokenizer(cfg["tokenizer"])
    out = {}
    specs = [(f"synth:{t}", "8k") for t in SYNTH_TASKS] + [(f"babilong:{t}", "8k") for t in ("qa1", "qa2", "qa3", "qa4", "qa5")]
    specs += [(f"infbench:{t}", "full") for t in cfg["infbench"]["tasks"]]
    for spec, L in specs:
        if spec.startswith("synth:"):
            path = data_file(spec.split(":")[1], L, 0, cfg)
            ex = next(r for i, r in enumerate(iter_jsonl(path)) if i == 3)
            ex["benchmark"] = "synth"
        else:
            ex = load_examples(spec, L, 1, 0, cfg)[0]
        system, before, after = prompt_for(ex, cfg)
        chat = chat_segments(tok, system, before, after, "<CONTEXT>")
        ev = sorted(ex.get("evidence") or [], key=lambda e: e["char_start"])
        out[spec] = {
            "id": ex["id"], "length": L, "length_tokens": ex["length_tokens"], "question": ex["question"],
            "answers": ex.get("answers"), "exclude": ex.get("exclude"), "scoring": ex.get("scoring", "official"),
            "depth_decile": ex.get("depth_decile"),
            "evidence": [{k: e.get(k) for k in ("text", "hop", "tok_start", "tok_end", "decile", "role")} for e in ev[:6]],
            "n_evidence": len(ev), "n_distractors": len(ex.get("distractors") or []),
            "distractors": [d["text"] for d in (ex.get("distractors") or [])[:4]],
            "excerpt": _excerpt(ex, ev[0] if ev else None),
            "prompt_prefix": chat.prefix, "prompt_suffix": chat.suffix,
        }
    return out


def main() -> int:
    cfg = load_task_config()
    synth = synth_summary(cfg)
    bab_len, locator, bab_rows = babilong_summary()
    inf, inf_rows = infbench_summary()
    sizes = json.loads((OUT / "sizes.json").read_text(encoding="utf-8"))
    figs = figures(bab_rows, inf_rows, cfg)
    ex = examples(cfg)
    from pcm.env import collect_env

    env = collect_env()
    atomic_write_json(OUT / "summary.json", {"code": {"git_commit": env["git_commit"], "git_dirty": env["git_dirty"]}, "synth": synth, "babilong_lengths": bab_len, "babilong_locator": locator,
                                             "infbench": inf, "sizes": sizes, "figures": figs})
    atomic_write_json(OUT / "examples.json", ex)

    print("## Synthetic cells (seed 0)")
    print("| task | length | target | n | min | median | max | within 1% | deciles | evidence items (median) | attempts | distinct first books |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in synth.items():
        t = s["tokens"]
        dec = "all 10 x10" if s["deciles"] and set(s["deciles"].values()) == {s["tokens"]["n"] // 10} and len(s["deciles"]) == 10 else (s["deciles"] or "none")
        print(f"| {s['task']} | {s['length']} | {s['target']} | {t['n']} | {t['min']} | {t['median']} | {t['max']} | {s['within_tolerance']} | {dec} | {s['evidence_items']['median']} | {s['attempts_mean']} | {s['first_books_distinct']} |")
    print("\n## BABILong lengths (qa1-qa10, 100 rows each)")
    print("| length | nominal | n | context min | median | max | median/nominal | prompt max | prompts <= 32,768 | YaRN factor for max prompt |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for L in cfg["babilong"]["lengths"]:
        s = bab_len[L]
        c, p = s["context_tokens"], s["prompt_tokens"]
        print(f"| {L} | {s['nominal'] or 0} | {c['n']} | {c['min']} | {c['median']} | {c['max']} | {s['ratio_median_to_nominal']} | {p['max']} | {s['prompts_within_32768']} | {s['yarn_factor_for_max_prompt']} |")
    print("\n## BABILong fact locator")
    print("| task | located | located that are 0k facts | share | aligned rows | 0k facts | matched | recall | rows exact |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in cfg["babilong"]["tasks"]:
        s = locator[t]
        print(f"| {t} | {s['located']} | {s['located_known']} | {s['share_located_that_are_0k_facts']} | {s.get('aligned_rows', 0)} | {s.get('facts_0k', 0)} | {s.get('matched', 0)} | {s['recall_vs_aligned_0k_row']} | {s.get('rows_exact', 0)} |")
    print("\n## InfiniteBench lengths")
    print("| task | n | context min | median | max | prompt max | > 128K | > 256K | <= 32,768 | YaRN factor | rows with evidence |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t, s in inf.items():
        c = s["context_tokens"]
        print(f"| {t} | {c['n']} | {c['min']} | {c['median']} | {c['max']} | {s['prompt_tokens']['max']} | {s['over_131072']} | {s['over_262144']} | {s['within_32768']} | {s['yarn_factor_for_max_prompt']} | {s['rows_with_evidence']} |")
    print("\n## Sizes on disk (bytes)")
    print(json.dumps(sizes, indent=1))
    print("\nfigures:", figs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
