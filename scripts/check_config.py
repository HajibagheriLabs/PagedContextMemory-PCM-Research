"""Check the shipped model configs against the geometry the specification assumes.

For each reader in configs/assets.yaml (check_config_models): assert layers,
heads, KV heads and head dimension; recompute KV bytes per token; count
parameters (from the weight files' headers, cross-checked against a count
rebuilt from the config) and report the bf16 weight size. Exits with status 1
if an assertion fails: the memory tables of the specification would then be wrong.

Use --models to check other models (for example on the server: Qwen/Qwen3-4B).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.hub import load_assets, model_path  # noqa: E402
from pcm.memory import GIB, KIB, kv_bytes_per_token, qwen3_param_count  # noqa: E402

ASSERTED = ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim")


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def stored_tensors(snapshot: Path) -> dict:
    """name -> {"shape", "dtype"} over every weight shard, from headers only."""
    tensors: dict = {}
    for shard in sorted(snapshot.glob("*.safetensors")):
        for name, meta in safetensors_header(shard).items():
            tensors[name] = {"shape": meta["shape"], "dtype": meta["dtype"]}
    return tensors


def numel(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def rope_info(config) -> dict:
    """rope_theta and scaling as the installed transformers exposes them."""
    params = getattr(config, "rope_parameters", None)
    if params:
        return {
            "api": "rope_parameters",
            "rope_theta": params.get("rope_theta"),
            "rope_type": params.get("rope_type"),
            "raw": dict(params),
        }
    return {
        "api": "rope_scaling",
        "rope_theta": getattr(config, "rope_theta", None),
        "rope_type": (getattr(config, "rope_scaling", None) or {}).get("type", "default"),
        "raw": getattr(config, "rope_scaling", None),
    }


def check_model(repo: str, expected: dict) -> dict:
    from transformers import AutoConfig

    snap = model_path(repo)
    config = AutoConfig.from_pretrained(snap)
    shipped = json.loads((snap / "config.json").read_text(encoding="utf-8"))

    failures = []
    geometry = {}
    for field in ASSERTED:
        got = getattr(config, field)
        geometry[field] = got
        if got != expected[field]:
            failures.append(f"{field}: expected {expected[field]}, got {got}")

    kv = kv_bytes_per_token(config.num_hidden_layers, config.num_key_value_heads, config.head_dim)
    if kv != expected["kv_bytes_per_token_bf16"]:
        failures.append(f"kv_bytes_per_token: expected {expected['kv_bytes_per_token_bf16']}, got {kv}")

    tensors = stored_tensors(snap)
    stored_total = sum(numel(t["shape"]) for t in tensors.values())
    tie = bool(getattr(config, "tie_word_embeddings", False))
    head_keys = [k for k in tensors if k.endswith("lm_head.weight")]
    duplicate_head = sum(numel(tensors[k]["shape"]) for k in head_keys) if tie else 0
    unique_params = stored_total - duplicate_head
    rebuilt = qwen3_param_count(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        attention_bias=bool(getattr(config, "attention_bias", False)),
        tie_word_embeddings=tie,
    )
    if rebuilt["total"] != unique_params:
        failures.append(
            f"parameter count: rebuilt from config {rebuilt['total']}, in weight files {unique_params}"
        )

    return {
        "repo": repo,
        "revision": snap.name,
        "geometry": geometry,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "max_position_embeddings_expected": expected["max_position_embeddings"],
        "max_position_embeddings_matches": config.max_position_embeddings
        == expected["max_position_embeddings"],
        "rope": rope_info(config),
        "shipped_config_rope_keys": {k: shipped[k] for k in shipped if k.startswith("rope")},
        "shipped_transformers_version": shipped.get("transformers_version"),
        "tie_word_embeddings": tie,
        "attention_bias": bool(getattr(config, "attention_bias", False)),
        "use_sliding_window": bool(getattr(config, "use_sliding_window", False)),
        "kv_bytes_per_token_bf16": kv,
        "kv_kib_per_token_bf16": kv / KIB,
        "params": {
            "total_unique": unique_params,
            "non_embedding": rebuilt["non_embedding"],
            "embedding": rebuilt["embedding"],
            "rebuilt_from_config_total": rebuilt["total"],
            "stored_in_weight_files": stored_total,
            "lm_head_stored_separately": bool(head_keys),
            "weight_dtypes": sorted({t["dtype"] for t in tensors.values()}),
        },
        "bf16_weight_bytes": unique_params * 2,
        "bf16_weight_gib": unique_params * 2 / GIB,
        "failures": failures,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", help="repo ids or keys; default from configs/assets.yaml")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "config_check.json")
    args = ap.parse_args()

    cfg = load_assets()
    expected = cfg["expected_geometry"]
    keys = args.models or cfg["check_config_models"]
    repos = [cfg["models"].get(k, k) for k in keys]

    import transformers

    results = [check_model(r, expected) for r in repos]
    report = {"transformers": transformers.__version__, "expected": expected, "models": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed = False
    for r in results:
        p = r["params"]
        print(f"== {r['repo']} (revision {r['revision'][:10]})")
        print(f"   geometry                  {r['geometry']}")
        print(f"   hidden_size               {r['hidden_size']}")
        print(f"   intermediate_size         {r['intermediate_size']}")
        print(f"   vocab_size                {r['vocab_size']}")
        print(f"   rope_theta                {r['rope']['rope_theta']}  (via {r['rope']['api']}, type {r['rope']['rope_type']})")
        note = "as expected" if r["max_position_embeddings_matches"] else "DIFFERS from expected, recorded"
        print(f"   max_position_embeddings   {r['max_position_embeddings']}  ({note}; original length for YaRN)")
        print(f"   KV bytes per token, bf16  {r['kv_bytes_per_token_bf16']:,.0f}  ({r['kv_kib_per_token_bf16']:.0f} KiB)")
        print(f"   parameters                {p['total_unique']:,}  (non-embedding {p['non_embedding']:,})")
        print(f"   stored in weight files    {p['stored_in_weight_files']:,}  (separate lm_head copy: {p['lm_head_stored_separately']})")
        print(f"   bf16 weight size          {r['bf16_weight_gib']:.2f} GiB")
        if r["failures"]:
            failed = True
            for f in r["failures"]:
                print(f"   ASSERTION FAILED: {f}")
        else:
            print("   all assertions passed")
    print(f"\nwrote {args.out}")
    if failed:
        print("STOP: the memory tables of the specification would be wrong for at least one model.")
        sys.exit(1)


if __name__ == "__main__":
    main()
