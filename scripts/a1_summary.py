"""Print the tables of reports/A1_engine.md from the files under results/A1_engine/."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "results" / "A1_engine"
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B"]


def g(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}g}"


def main() -> None:
    tests = {m: json.loads((OUT / f"tests_{m}.json").read_text(encoding="utf-8")) for m in MODELS}
    t = {m: tests[m]["tests"] for m in MODELS}
    print("text:", {m: tests[m]["text"]["book"] for m in MODELS})
    for m in MODELS:
        stamps = {(v.get("run", {}).get("git_commit", "unstamped") or "none")[:10] + ("+dirty" if v.get("run", {}).get("git_dirty", True) else "") for v in t[m].values()}
        print(f"{m}: {len(t[m])} test records, code stamps {sorted(stamps)}")

    def row(label, fn):
        print(f"| {label} | " + " | ".join(fn(t[m]) for m in MODELS) + " |")

    print("\n## T1")
    row("f32 last-logit max abs diff", lambda x: g(x["T1_f32"]["last_logit_max_abs_diff"]))
    row("f32 largest |logit|", lambda x: g(x["T1_f32"]["logit_abs_max"]))
    row("f32 cache K max abs diff", lambda x: g(x["T1_f32"]["cache_key_max_abs_diff"]))
    row("f32 cache V max abs diff", lambda x: g(x["T1_f32"]["cache_value_max_abs_diff"]))
    row("bf16 top-1 one-shot / chunked", lambda x: f"{x['T1_bf16']['top1_one_shot']!r} / {x['T1_bf16']['top1_chunked']!r}")
    row("bf16 last-logit max abs diff", lambda x: g(x["T1_bf16"]["last_logit_max_abs_diff"]))

    print("\n## T2")
    row("K max abs diff (spec < 1e-3)", lambda x: g(x["T2"]["key_max_abs_diff"]))
    row("V max abs diff (spec < 1e-3)", lambda x: g(x["T2"]["value_max_abs_diff"]))
    row("layers with K diff < 1e-3", lambda x: str(sum(d < 1e-3 for d in x["T2"]["key_max_abs_diff_per_layer"])) + " of 28")
    row("layers with V diff < 1e-3", lambda x: str(sum(d < 1e-3 for d in x["T2"]["value_max_abs_diff_per_layer"])) + " of 28")
    row("largest |K| (min..max over layers)", lambda x: f"{g(min(x['T2']['key_abs_max_per_layer']))} .. {g(max(x['T2']['key_abs_max_per_layer']))}")
    row("largest |V| (min..max over layers)", lambda x: f"{g(min(x['T2']['value_abs_max_per_layer']))} .. {g(max(x['T2']['value_abs_max_per_layer']))}")
    row("K max abs diff without re-basing", lambda x: g(x["T2"]["key_max_abs_diff_without_rebasing"]))
    row("K max rel diff", lambda x: g(x["T2_relative"]["key_max_rel_diff"]))
    row("V max rel diff", lambda x: g(x["T2_relative"]["value_max_rel_diff"]))
    row("logit diff via re-based vs direct", lambda x: g(x["T2_relative"]["last_logit_max_abs_diff"]))
    row("logit diff with no re-basing", lambda x: g(x["T2_relative"]["last_logit_max_abs_diff_without_rebasing"]))
    row("same top-1", lambda x: str(x["T2_relative"]["same_top1"]))
    row("f64 model, library f32 angles: K max abs", lambda x: g(x["T2_float64"]["float64_model_float32_angles_key_max_abs_diff"]))
    row("f64 model, library f32 angles: V max abs", lambda x: g(x["T2_float64"]["float64_model_float32_angles_value_max_abs_diff"]))
    row("f64 model, f64 angles: K max abs", lambda x: g(x["T2_float64"]["key_max_abs_diff"]))
    row("f64 model, f64 angles: V max abs", lambda x: g(x["T2_float64"]["value_max_abs_diff"]))
    row("f64 model, f64 angles: K max rel", lambda x: g(x["T2_float64"]["key_max_rel_diff"]))
    row("f64 model, f64 angles: V max rel", lambda x: g(x["T2_float64"]["value_max_rel_diff"]))
    row("f64/f64: layers with K abs diff < 1e-9", lambda x: str(sum(d < 1e-9 for d in x["T2_float64"]["key_max_abs_diff_per_layer"])) + " of 28")
    for m in MODELS:
        print(m, "f64 K per layer:", [g(d, 2) for d in t[m]["T2_float64"]["key_max_abs_diff_per_layer"]])

    print("\n## T3")
    for k in ("prefix_tokens", "page_tokens", "suffix_tokens", "total_tokens_kv", "total_tokens_text", "question_positions_kv", "question_positions_text", "logits_finite"):
        row(k, lambda x, k=k: str(x["T3"][k]))
    row("layer-0 K max rel diff", lambda x: g(x["T3"]["layer0_key_max_rel_diff"]))
    row("layer-0 V max rel diff", lambda x: g(x["T3"]["layer0_value_max_rel_diff"]))
    row("first page K max rel diff, all layers", lambda x: g(x["T3"]["first_page_key_max_rel_diff_all_layers"]))
    row("second page K max rel diff, layer 0", lambda x: g(x["T3"]["second_page_key_max_rel_diff_per_layer"][0]))
    row("second page K Frobenius rel diff, layers 1..27 (min..max)", lambda x: f"{g(min(x['T3']['second_page_key_frobenius_rel_diff_per_layer'][1:]))} .. {g(max(x['T3']['second_page_key_frobenius_rel_diff_per_layer'][1:]))}")
    row("last-logit max abs diff KV vs text", lambda x: g(x["T3"]["last_logit_max_abs_diff_kv_vs_text"]))
    row("top-1 KV / text", lambda x: f"{x['T3']['top1_kv']!r} / {x['T3']['top1_text']!r}")

    print("\n## T4")
    for c in ("native_16k", "no_yarn_48k", "yarn_48k", "yarn_100k"):
        row(c + " tokens", lambda x, c=c: str(x["T4"]["conditions"][c]["tokens"]))
        row(c + " loss tail", lambda x, c=c: f"{x['T4']['conditions'][c]['loss_tail']:.3f}")
        row(c + " loss all", lambda x, c=c: f"{x['T4']['conditions'][c]['loss_all']:.3f}")
        row(c + " peak GiB", lambda x, c=c: f"{x['T4']['conditions'][c]['peak_vram_gib']:.2f}")
    row("yarn rope", lambda x: json.dumps({k: x["T4"]["rope_yarn"][k] for k in ("rope_type", "factor", "original_max_position_embeddings", "max_position_embeddings", "rope_theta")}))
    row("yarn attention_scaling", lambda x: g(x["T4"]["rope_yarn"]["attention_scaling"], 5))

    print("\n## T5")
    for k in ("allocated_before_gib", "cache_gib", "peak_vram_gib", "peak_reserved_gib", "tokens_per_second"):
        row(k, lambda x, k=k: f"{x['T5'][k]:.2f}")
    row("generate_greedy", lambda x: f"{x['generate_greedy']['answer']!r} matches={x['generate_greedy']['matches_library_generate']}")

    print("\n## Throughput")
    all_rows = [json.loads(line) for line in (OUT / "throughput.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # The file is append-only and may hold rows from several commits: report the most recent clean one.
    clean = [r for r in all_rows if r.get("git_dirty") is False]
    commit = clean[-1]["git_commit"] if clean else None
    rows = [r for r in clean if r["git_commit"] == commit] if clean else all_rows
    print(f"rows in file: {len(all_rows)}; reported: {len(rows)} from commit {str(commit)[:10]}")
    print("| model | tokens | runs | median tok/s | min..max tok/s | median s | peak GiB | reserved GiB |")
    for model in sorted({r["model"] for r in rows}):
        for n in sorted({r["tokens"] for r in rows}):
            sel = [r for r in rows if r["model"] == model and r["tokens"] == n]
            tps = [r["tokens_per_second"] for r in sel]
            print(
                f"| {model} | {n:,} | {len(sel)} | {statistics.median(tps):,.0f} | {min(tps):,.0f}..{max(tps):,.0f} | "
                f"{statistics.median(r['seconds'] for r in sel):.2f} | {max(r['peak_vram_gib'] for r in sel):.2f} | "
                f"{max(r['peak_reserved_gib'] for r in sel):.2f} |"
            )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
