"""Smoke test: the reader answers one question and the embedder embeds two sentences.

Loads the primary reader in bf16 on the GPU with SDPA attention, applies the
chat template with thinking disabled, generates greedily, then loads the
embedder, prints the cosine similarity of two sentences and the peak VRAM.
Writes reports/smoke.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.hub import load_assets, model_path  # noqa: E402

GIB = 1024**3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", default="reader", help="key in configs/assets.yaml or repo id")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "smoke.json")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_assets()
    smoke = cfg["smoke"]
    reader_repo = cfg["models"].get(args.reader, args.reader)
    embedder_repo = cfg["models"]["embedder"]

    if not torch.cuda.is_available():
        print("STOP: no CUDA device.")
        sys.exit(1)
    device = torch.device("cuda:0")
    free0, total = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)

    # --- reader ---
    reader_dir = model_path(reader_repo)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(reader_dir)
    model = AutoModelForCausalLM.from_pretrained(
        reader_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    load_s = time.perf_counter() - t0

    messages = [{"role": "user", "content": smoke["question"]}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=smoke["max_new_tokens"],
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    gen_s = time.perf_counter() - t0
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    reader_peak = torch.cuda.max_memory_allocated()
    param = next(model.parameters())
    reader_info = {
        "repo": reader_repo,
        "revision": reader_dir.name,
        "dtype": str(param.dtype),
        "device": str(param.device),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights_gib_on_gpu": sum(p.numel() * p.element_size() for p in model.parameters()) / GIB,
        "load_seconds": round(load_s, 2),
        "prompt": prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "thinking_enabled": False,
        "decoding": "greedy",
        "max_new_tokens": smoke["max_new_tokens"],
        "new_tokens": int(new_tokens.shape[0]),
        "answer": answer,
        "generate_seconds": round(gen_s, 2),
        "peak_vram_gib_reader_only": reader_peak / GIB,
    }
    print(f"reader      {reader_repo}  {reader_info['dtype']}  attn={reader_info['attn_implementation']}")
    print(f"prompt      {prompt!r}")
    print(f"answer      {answer!r}")

    # --- embedder (reader stays loaded: the PCM text-mode layout) ---
    from sentence_transformers import SentenceTransformer

    embedder_dir = model_path(embedder_repo)
    embedder = SentenceTransformer(
        str(embedder_dir), device="cuda", model_kwargs={"dtype": torch.bfloat16}
    )
    sentences = smoke["embed_sentences"]
    with torch.inference_mode():
        emb = embedder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    emb = emb.float()
    cosine = float((emb[0] * emb[1]).sum())
    embed_info = {
        "repo": embedder_repo,
        "revision": embedder_dir.name,
        "sentences": sentences,
        "dimension": int(emb.shape[1]),
        "cosine_similarity": cosine,
    }
    print(f"embedder    {embedder_repo}  dim={embed_info['dimension']}")
    print(f"cosine      {cosine:.4f}")

    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    print(f"peak VRAM   {peak / GIB:.2f} GiB allocated, {reserved / GIB:.2f} GiB reserved (reader + embedder)")

    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_gib": total / GIB,
        "vram_free_before_gib": free0 / GIB,
        "reader": reader_info,
        "embedder": embed_info,
        "peak_vram_gib_allocated": peak / GIB,
        "peak_vram_gib_reserved": reserved / GIB,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
