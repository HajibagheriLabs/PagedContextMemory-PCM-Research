"""Run A7 (configs/a7_codecs.yaml): E7, store codecs. Resumable: finished rows are skipped.

    python scripts/run_a7.py --stage pages     # PG-19 pages: 64 calibration (validation), 256 held-out (test)
    python scripts/run_a7.py --stage fit       # low-rank bases from the calibration pages
    python scripts/run_a7.py --stage part1     # every lossy codec on the held-out pages (GPU)
    python scripts/run_a7.py --stage part2     # lossless coders (CPU)
    python scripts/run_a7.py --stage timing    # encode and decode ms per page, on an otherwise idle GPU
    python scripts/run_a7.py --stage part4     # a 256K document's store: sizes, VRAM fit, expansion latency
    python scripts/run_a7.py --stage all       # the five in order
    python scripts/run_a7.py --stage part3 --pilot             # Part 3 cells: pilot rows and projection
    python scripts/run_a7.py --stage part3 --max-hours 8       # Part 3: KV mode with each codec, paired with bf16

Part 3 (task accuracy in KV mode, D-050) needs E6's tuned fix (results/E6_kv/ape_choice.json) and is run only
if Gate G2 passed; otherwise the report gives Part 1's fidelity measures in its place.
Results go to results/E7_codecs/; page token ids and the fitted bases to data/cache/codecs/ (git-ignored).
A lost GPU stops the run with exit status 75; rerun the same command once the GPU is back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pcm.env import collect_env  # noqa: E402
from pcm.health import EXIT_GPU_LOST, GpuTelemetry, classify_failure, interruption_record, preflight  # noqa: E402
from pcm.runlog import JsonlLog, atomic_write_bytes, atomic_write_json  # noqa: E402
from pcm.tasks.common import load_task_config  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "a7_codecs.yaml"
OUT = REPO_ROOT / "results" / "E7_codecs"
STAGES = ("pages", "fit", "part1", "part2", "timing", "part4")
EXP3 = "E7_codecs"
GIB = 1024**3


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    p = REPO_ROOT / "logs" / "A7" / "run.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d ") + line + "\n")


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------- shared pieces
class Ctx:
    """Model, tokenizer and the shared prefix, loaded once."""

    def __init__(self, cfg: dict):
        from pcm.eval import runner as R
        from pcm.models import chat_segments, encode_text, load_reader
        from pcm.prefill import prefill_in_blocks
        from pcm.store.kv import rope_inv_freq

        self.cfg = cfg
        self.tcfg = load_task_config()
        self.model, self.tok = load_reader(cfg["reader"], dtype=getattr(torch, cfg["dtype"]))
        pf = cfg["prefix_from"]
        ex = R.load_examples(pf["task"], pf["length"], 1, int(cfg["seed"]), self.tcfg)[0]
        system, before, after = R.prompt_for(ex, self.tcfg)
        chat = chat_segments(self.tok, system, before, after, ex["context"])
        self.prefix_ids = encode_text(self.tok, chat.prefix)
        self.P = prefill_in_blocks(self.model, self.prefix_ids)
        self.start = int(len(self.prefix_ids))
        self.inv_freq = rope_inv_freq(self.model)

    def encode(self, ids):
        from pcm.store.kv import encode_page

        return encode_page(self.model, self.P, torch.as_tensor(ids, dtype=torch.long))

    @torch.no_grad()
    def continuation_logprobs(self, page_cache, cont_ids) -> torch.Tensor:
        """Log-probabilities [len(cont), vocab] (float32) of the continuation tokens fed after P + page."""
        from pcm.store.kv import KVCache

        cache = KVCache.concat([self.P, page_cache])
        n = cache.n_tokens
        q = torch.as_tensor(cont_ids, dtype=torch.long, device=self.model.device)[None]
        lib = cache.to_library(self.model.config, capacity=n + q.shape[1])
        pos = torch.arange(n, n + q.shape[1], device=self.model.device)[None]
        out = self.model(input_ids=q, position_ids=pos, past_key_values=lib, use_cache=True)
        return torch.log_softmax(out.logits[0].float(), dim=-1)


def codec_names(cfg: dict) -> list[str]:
    p = cfg["part1"]
    names = ["bf16"] + [f"int{b}-{g}" for b in p["quant_bits"] for g in p["granularities"]]
    names += [f"lowrank{r}-int{b}" for r in p["lowrank_ranks"] for b in p["lowrank_bits"]]
    return names


def basis_path(cfg: dict) -> Path:
    return REPO_ROOT / cfg["basis_dir"] / "basis.pt"


def pages_path(cfg: dict) -> Path:
    return REPO_ROOT / cfg["basis_dir"] / "pages.json"


def require_clean() -> dict:
    env = collect_env()
    if env["git_dirty"]:
        raise SystemExit("uncommitted code: commit first (rows must carry a clean commit)")
    return env


# ---------------------------------------------------------------- stage: pages
def stage_pages(cfg: dict) -> int:
    pc = cfg["pages"]
    roots = sorted((REPO_ROOT / pc["book_cache"]).glob("*/index.json"))
    if len(roots) != 1:
        raise SystemExit(f"expected one book cache, found {len(roots)}")
    root = roots[0].parent
    books = sorted(json.loads(roots[0].read_text(encoding="utf-8"))["books"], key=lambda b: b["key"])
    from transformers import AutoTokenizer

    from pcm.hub import model_path

    tok = AutoTokenizer.from_pretrained(model_path(cfg["reader"]))

    def windows(split, n, need, per_book, stride):
        out = []
        for b in books:
            if b["split"] != split:
                continue
            offsets = np.load(root / f"{b['key']}.npy")
            text = (root / f"{b['key']}.txt").read_text(encoding="utf-8")
            ntok = len(offsets) - 1
            for k in range(per_book):
                a = int(pc["skip_tokens"]) + k * int(stride)
                if a + need + 64 > ntok:
                    break
                chars = text[int(offsets[a]):int(offsets[a + need + 64])]
                ids = tok(chars, add_special_tokens=False)["input_ids"][:need]
                if len(ids) == need:
                    out.append({"book": b["key"], "token_offset": a, "ids": [int(x) for x in ids]})
                if len(out) == n:
                    return out
        raise SystemExit(f"only {len(out)} {split} windows of {need} tokens; need {n}")

    cal = pc["calibration"]
    held = pc["heldout"]
    calib = windows(cal["split"], int(cal["n"]), int(cal["page_tokens"]), int(cal["per_book"]), int(cal.get("stride", 0)))
    test = windows(held["split"], int(held["n"]), int(held["page_tokens"]) + int(held["continuation"]), int(held["per_book"]), int(held["stride"]))
    assert not ({p["book"] for p in calib} & {p["book"] for p in test})
    data = {"calibration": calib, "heldout": test}
    blob = json.dumps(data).encode("utf-8")
    atomic_write_bytes(pages_path(cfg), blob)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {k: [{"book": p["book"], "token_offset": p["token_offset"], "tokens": len(p["ids"])} for p in v] for k, v in data.items()}
    summary["sha256_of_ids"] = hashlib.sha256(blob).hexdigest()
    summary["books"] = {k: len({p["book"] for p in v}) for k, v in data.items()}
    atomic_write_json(OUT / "pages.json", summary)
    log(f"pages: {len(calib)} calibration windows from {summary['books']['calibration']} validation books, "
        f"{len(test)} held-out windows from {summary['books']['heldout']} test books")
    return 0


def load_pages(cfg: dict) -> dict:
    p = pages_path(cfg)
    if not p.is_file():
        raise SystemExit("run --stage pages first")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- stage: fit
def stage_fit(cfg: dict, ctx: Ctx) -> int:
    from pcm.store.codecs import LowRankBasis

    env = require_clean()
    pages = load_pages(cfg)["calibration"]
    t0 = time.perf_counter()
    basis = LowRankBasis.fit(((ctx.encode(p["ids"]), ctx.start) for p in pages), ctx.inv_freq, device=ctx.model.device)
    fit_s = time.perf_counter() - t0
    path = basis_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    basis.save(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rec = {"git_commit": env["git_commit"], "git_dirty": env["git_dirty"], "tokens": basis.tokens, "pages": len(pages), "fit_s": fit_s,
           "prefix_tokens": ctx.start, "file": str(path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha, "explained": {}}
    for name, var in (("k", basis.k_var), ("v", basis.v_var)):
        tot = var.clamp_min(0).sum(dim=1)
        for r in cfg["part1"]["lowrank_ranks"]:
            share = (var[:, :r].clamp_min(0).sum(dim=1) / tot).tolist()
            rec["explained"][f"{name}|{r}"] = {"mean": float(np.mean(share)), "min": float(np.min(share)), "per_layer": [round(x, 5) for x in share]}
    atomic_write_json(OUT / "basis.json", rec)
    log(f"fit: {basis.tokens} tokens in {fit_s:.1f} s; explained variance at r=128/256/512: keys "
        + "/".join(f"{rec['explained'][f'k|{r}']['mean']:.3f}" for r in (128, 256, 512))
        + ", values " + "/".join(f"{rec['explained'][f'v|{r}']['mean']:.3f}" for r in (128, 256, 512)))
    return 0


def load_basis(cfg: dict, device):
    from pcm.store.codecs import LowRankBasis

    path = basis_path(cfg)
    if not path.is_file():
        raise SystemExit("run --stage fit first")
    rec = json.loads((OUT / "basis.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != rec["sha256"]:
        raise SystemExit("the basis file differs from the one recorded in basis.json")
    return LowRankBasis.load(path, device=device)


# ---------------------------------------------------------------- stage: part1
def stage_part1(cfg: dict, ctx: Ctx) -> int:
    from pcm.store import codecs as C

    env = require_clean()
    pages = load_pages(cfg)["heldout"]
    basis = load_basis(cfg, ctx.model.device)
    names = codec_names(cfg)
    codecs = {n: C.make_codec(n, basis, ctx.inv_freq, int(cfg["group"])) for n in names}
    out = JsonlLog(OUT / "part1.jsonl")
    done = out.done_ids()
    L = int(cfg["pages"]["heldout"]["page_tokens"])
    # warm-up (kernels, allocator), not recorded
    w = ctx.encode(pages[0]["ids"][:L])
    for c in codecs.values():
        c.decode(c.encode(w, ctx.start))
    sync()
    for i, p in enumerate(pages):
        todo = [n for n in names + (["text"] if cfg["part1"]["text"] else []) if f"{i}|{n}" not in done]
        if not todo:
            continue
        ids, cont = p["ids"][:L], p["ids"][L:]
        ref = ctx.encode(ids)
        ref_lp = ctx.continuation_logprobs(ref, cont)
        ref_top = ref_lp.argmax(-1)
        for n in todo:
            if n == "text":
                codec = C.TextCodec(ctx.model, ctx.P)
                sync()
                t = time.perf_counter()
                enc = codec.encode_ids(ids, ctx.start)
                enc_ms = 1000 * (time.perf_counter() - t)
            else:
                codec = codecs[n]
                sync()
                t = time.perf_counter()
                enc = codec.encode(ref, ctx.start)
                sync()
                enc_ms = 1000 * (time.perf_counter() - t)
            t = time.perf_counter()
            dec = codec.decode(enc)
            sync()
            dec_ms = 1000 * (time.perf_counter() - t)
            err = C.relative_error(ref, dec)
            lp = ctx.continuation_logprobs(dec, cont)
            kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1)
            row = {"id": f"{i}|{n}", "page": i, "book": p["book"], "codec": n, "tokens": L, "bytes": int(enc.nbytes),
                   "bytes_per_token": enc.nbytes / L, "ratio": (ref.nbytes / enc.nbytes), "shared_bytes": int(getattr(codec, "shared_nbytes", 0)),
                   "encode_ms": enc_ms, "decode_ms": dec_ms, "rel_err_k": err["k"], "rel_err_v": err["v"],
                   "kl": float(kl.mean()), "kl_max": float(kl.max()), "top1": float((lp.argmax(-1) == ref_top).float().mean()),
                   "git_commit": env["git_commit"], "git_dirty": env["git_dirty"]}
            out.append(row)
            del dec, lp
        if i % 16 == 0:
            log(f"part1: page {i + 1}/{len(pages)}")
        del ref, ref_lp
    log("part1: done")
    return 0


# ---------------------------------------------------------------- stage: timing
def stage_timing(cfg: dict, ctx: Ctx) -> int:
    """Encode and decode ms per page of every Part 1 codec (median of 5 repeats per page) on the first 32 held-out
    pages, run with nothing else on the GPU (D-049: Part 1's fidelity pass shared it with A6 tuning)."""
    from pcm.store import codecs as C

    env = require_clean()
    pages = load_pages(cfg)["heldout"][:32]
    basis = load_basis(cfg, ctx.model.device)
    names = codec_names(cfg)
    codecs = {n: C.make_codec(n, basis, ctx.inv_freq, int(cfg["group"])) for n in names}
    out = JsonlLog(OUT / "timing.jsonl")
    done = out.done_ids()
    L = int(cfg["pages"]["heldout"]["page_tokens"])
    w = ctx.encode(pages[0]["ids"][:L])
    for c in codecs.values():
        c.decode(c.encode(w, ctx.start))
    sync()
    for i, p in enumerate(pages):
        ids = p["ids"][:L]
        ref = ctx.encode(ids)
        for n in names + ["text"]:
            if f"{i}|{n}" in done:
                continue
            enc_t, dec_t = [], []
            for _ in range(5):
                if n == "text":
                    codec = C.TextCodec(ctx.model, ctx.P)
                    sync()
                    t = time.perf_counter()
                    enc = codec.encode_ids(ids, ctx.start)
                else:
                    codec = codecs[n]
                    sync()
                    t = time.perf_counter()
                    enc = codec.encode(ref, ctx.start)
                sync()
                enc_t.append(1000 * (time.perf_counter() - t))
                t = time.perf_counter()
                dec = codec.decode(enc)
                sync()
                dec_t.append(1000 * (time.perf_counter() - t))
                del dec
            out.append({"id": f"{i}|{n}", "page": i, "codec": n, "encode_ms": statistics.median(enc_t), "decode_ms": statistics.median(dec_t),
                        "git_commit": env["git_commit"], "git_dirty": env["git_dirty"]})
        del ref
    log("timing: done")
    return 0


# ---------------------------------------------------------------- stage: part2
def doc_for_part4(cfg: dict, tok, tcfg):
    """(example, prefix ids, units, page bounds) of the Part 4 document, cut as configs/pcm_text.yaml cuts it."""
    from pcm.chunking import cut_document, page_bounds, units_per_page
    from pcm.eval import runner as R
    from pcm.models import chat_segments, encode_text

    pt = yaml.safe_load((REPO_ROOT / "configs" / "pcm_text.yaml").read_text(encoding="utf-8"))["pcm_text"]
    d = cfg["part4"]["document"]
    ex = R.load_examples(d["task"], d["length"], int(d["index"]) + 1, int(cfg["seed"]), tcfg)[int(d["index"])]
    system, before, after = R.prompt_for(ex, tcfg)
    chat = chat_segments(tok, system, before, after, ex["context"])
    units = cut_document(tok, chat.context, int(pt["unit"]), int(pt["snap"]))
    pb = page_bounds(units.bounds, units_per_page(int(pt["page"]), int(pt["unit"])))
    return ex, chat, encode_text(tok, chat.prefix), encode_text(tok, chat.suffix), units, pb, pt


def stage_part2(cfg: dict, ctx: Ctx) -> int:
    from pcm.store import codecs as C

    env = require_clean()
    pages = load_pages(cfg)["heldout"][: int(cfg["part2"]["pages"])]
    out = JsonlLog(OUT / "part2.jsonl")
    done = out.done_ids()
    coders = cfg["part2"]["coders"]
    L = int(cfg["pages"]["heldout"]["page_tokens"])

    def measure(item_id, raw, coder_name, extra):
        if f"{item_id}|{coder_name}" in done:
            return
        comp, decomp = C.coder(coder_name)
        t = time.perf_counter()
        blob = comp(raw)
        cs = time.perf_counter() - t
        t = time.perf_counter()
        back = decomp(blob)
        ds = time.perf_counter() - t
        if back != raw:
            raise RuntimeError(f"{coder_name} did not round-trip {item_id}")
        mb = len(raw) / 1e6
        out.append({"id": f"{item_id}|{coder_name}", "item": item_id.split("|", 1)[1], "coder": coder_name, "raw_bytes": len(raw),
                    "compressed_bytes": len(blob), "ratio": len(raw) / len(blob), "compress_s": cs, "decompress_s": ds,
                    "compress_mb_s": mb / cs if cs > 0 else None, "decompress_mb_s": mb / ds if ds > 0 else None,
                    "git_commit": env["git_commit"], "git_dirty": env["git_dirty"], **extra})
        log(f"part2 {item_id} {coder_name}: ratio {len(raw) / len(blob):.3f}, {mb / cs:.1f} / {mb / ds:.1f} MB/s")

    for i, p in enumerate(pages):
        ref = ctx.encode(p["ids"][:L])
        bf = C.BF16()
        raw = bf.to_bytes(bf.encode(ref, ctx.start))
        t = time.perf_counter()
        shuf = C.shuffle_bytes(raw)
        shuffle_s = time.perf_counter() - t
        items = {"bf16": (raw, {}), "bf16-shuffle": (shuf, {"shuffle_s": shuffle_s})}
        for spec in cfg["part2"]["codecs"]:
            c = C.make_codec(spec, group=int(cfg["group"]))
            items[spec] = (c.to_bytes(c.encode(ref, ctx.start)), {})
        del ref
        for name, (data, extra) in items.items():
            for cn in coders:
                if cn not in cfg["part2"].get("kv_skip_coders", []):
                    measure(f"{i}|{name}", data, cn, {"page": i, **extra})
    # token ids (uint32) and UTF-8 text of the Part 4 document
    ex, chat, _, _, units, _, _ = doc_for_part4(cfg, ctx.tok, ctx.tcfg)
    ids = np.asarray(units.ids, dtype="<u4").tobytes()
    text = chat.context.encode("utf-8")
    for name, data in (("ids-uint32", ids), ("text-utf8", text)):
        for cn in coders:
            measure(f"doc|{name}", data, cn, {"tokens": int(units.n_tokens), "example": ex["id"]})
    log("part2: done")
    return 0


def choose_coder(cfg: dict) -> dict:
    """The coder for int4 + lossless, by the rule of the config (written once)."""
    path = OUT / "coder_choice.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in JsonlLog(OUT / "part2.jsonl").rows() if r["item"] == "int4"]
    if not rows:
        raise SystemExit("run --stage part2 first")
    agg = {}
    for cn in cfg["part2"]["coders"]:
        rs = [r for r in rows if r["coder"] == cn]
        raw, comp = sum(r["raw_bytes"] for r in rs), sum(r["compressed_bytes"] for r in rs)
        agg[cn] = {"ratio": raw / comp, "decompress_mb_s": raw / 1e6 / sum(r["decompress_s"] for r in rs),
                   "compress_mb_s": raw / 1e6 / sum(r["compress_s"] for r in rs)}
    ok = [c for c, a in agg.items() if a["decompress_mb_s"] >= float(cfg["part2"]["min_decompress_mb_s"])]
    best = max(ok, key=lambda c: (round(agg[c]["ratio"], 4), agg[c]["decompress_mb_s"])) if ok else None
    rec = {"rule": "highest ratio on the int4 payload among coders decompressing at >= "
                   f"{cfg['part2']['min_decompress_mb_s']} MB/s; ties: faster", "coders": agg, "choice": best}
    atomic_write_json(path, rec)
    return rec


def choose_lowrank(cfg: dict) -> dict:
    """The best low-rank setting, by the rule of the config (written once)."""
    path = OUT / "lowrank_choice.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = list(JsonlLog(OUT / "part1.jsonl").rows())
    by = {}
    for r in rows:
        by.setdefault(r["codec"], []).append(r)
    agg = {c: {"kl": float(np.mean([r["kl"] for r in rs])), "bytes_per_token": float(np.mean([r["bytes_per_token"] for r in rs])), "n": len(rs)}
           for c, rs in by.items()}
    ref = agg["int4-kcvt"]["kl"]
    lr = [c for c in agg if c.startswith("lowrank")]
    ok = [c for c in lr if agg[c]["kl"] <= ref]
    best = min(ok, key=lambda c: (agg[c]["bytes_per_token"], agg[c]["kl"])) if ok else min(lr, key=lambda c: agg[c]["kl"])
    rec = {"rule": "smallest bytes per token among low-rank codecs with mean KL <= int4 (KIVI); none: the lowest KL",
           "int4_kl": ref, "lowrank": {c: agg[c] for c in lr}, "choice": best, "qualified": bool(ok)}
    atomic_write_json(path, rec)
    return rec


# ---------------------------------------------------------------- stage: part4
def stage_part4(cfg: dict, ctx: Ctx) -> int:
    from pcm.expand import expand_kv, expand_text
    from pcm.prefill import generate_greedy, prefill_in_blocks
    from pcm.store import codecs as C
    from pcm.store.kv import encode_page

    env = require_clean()
    out = JsonlLog(OUT / "part4.jsonl")
    done = out.done_ids()
    basis = load_basis(cfg, ctx.model.device)
    lowrank = choose_lowrank(cfg)["choice"]
    coder_rec = choose_coder(cfg)
    names = ["bf16", "int8", "int4", "int2", lowrank]
    if coder_rec["choice"]:
        names.append(f"int4+{coder_rec['choice']}")
    codecs = {n: C.make_codec(n, basis, ctx.inv_freq, int(cfg["group"])) for n in names}
    ex, chat, prefix_ids, suffix_ids, units, pb, pt = doc_for_part4(cfg, ctx.tok, ctx.tcfg)
    model = ctx.model
    P = prefill_in_blocks(model, prefix_ids)
    start = int(len(prefix_ids))
    ids = torch.as_tensor(units.ids.astype(np.int64))
    n_pages = len(pb) - 1
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # (1) store size per codec, every page of the document
    if "sizes" not in done:
        total = {n: 0 for n in names}
        enc_s = {n: 0.0 for n in names}
        sync()
        t0 = time.perf_counter()
        for j in range(n_pages):
            a, b = int(pb[j]), int(pb[j + 1])
            cache = encode_page(model, P, ids[a:b])
            for n, c in codecs.items():
                t = time.perf_counter()
                total[n] += c.encode(cache, start).nbytes
                sync()
                enc_s[n] += time.perf_counter() - t
            del cache
        sync()
        ingest_s = time.perf_counter() - t0
        text_bytes = int(units.n_tokens) * 4
        rec = {"id": "sizes", "example": ex["id"], "tokens": int(units.n_tokens), "pages": n_pages, "page": int(pt["page"]),
               "ingest_s_all_codecs": ingest_s, "weights_bytes": weights_bytes, "codecs": {}}
        for n in names:
            shared = int(codecs[n].shared_nbytes)
            rec["codecs"][n] = {"store_bytes": total[n], "shared_bytes": shared, "bytes_per_token": (total[n] + shared) / units.n_tokens,
                                "encode_s": enc_s[n]}
        rec["codecs"]["text"] = {"store_bytes": text_bytes, "shared_bytes": 0, "bytes_per_token": 4.0, "encode_s": 0.0}
        rec.update({"git_commit": env["git_commit"], "git_dirty": env["git_dirty"]})
        out.append(rec)
        log("part4 sizes: " + ", ".join(f"{n} {v['bytes_per_token'] * units.n_tokens / GIB:.2f} GiB" for n, v in rec["codecs"].items()))

    # (2) expansion latency: a full budget of pages (28,672 tokens), decompressed and stitched vs recomputed
    per = int(pb[1] - pb[0])
    k = int(cfg["part4"]["budget_tokens"]) // int(pt["page"])
    sel = list(range(0, n_pages, max(1, n_pages // k)))[:k]   # spread over the document: every boundary is a gap
    from pcm.models import encode_text

    gap_ids = encode_text(ctx.tok, pt["gap_marker"])
    G = encode_page(model, P, gap_ids)
    spans = [(int(pb[j]), int(pb[j + 1])) for j in sel]
    body, _ = expand_text(ids, spans, gap_ids)
    for rep in range(3):
        rid = f"latency|{rep}"
        if rid in done:
            continue
        rec = {"id": rid, "pages": len(sel), "context_tokens": int(len(body)), "codecs": {}}
        caches = [encode_page(model, P, ids[a:b]) for a, b in spans]
        for n, c in codecs.items():
            encs = [c.encode(x, start) for x in caches]
            sync()
            t = time.perf_counter()
            segs = []
            for j, e in enumerate(encs):
                if j:
                    segs.append(G)
                segs.append(c.decode(e))
            cache, n_ctx = expand_kv(P, segs, ctx.inv_freq)
            sync()
            rec["codecs"][n] = {"expand_ms": 1000 * (time.perf_counter() - t)}
            if n_ctx != len(body):
                raise RuntimeError("stitched context differs from the text context")
            del encs, segs, cache
        sync()
        t = time.perf_counter()
        segs = []
        for j, (a, b) in enumerate(spans):
            if j:
                segs.append(G)
            segs.append(encode_page(model, P, ids[a:b]))
        cache, _ = expand_kv(P, segs, ctx.inv_freq)
        sync()
        rec["codecs"]["text"] = {"expand_ms": 1000 * (time.perf_counter() - t), "what": "each page re-encoded after P, then stitched"}
        del segs, cache
        sync()
        t = time.perf_counter()
        seq = prefill_in_blocks(model, torch.cat([prefix_ids, body]))
        sync()
        rec["recompute_text_mode_ms"] = 1000 * (time.perf_counter() - t)
        del seq, caches
        rec.update({"git_commit": env["git_commit"], "git_dirty": env["git_dirty"]})
        out.append(rec)
        log(f"part4 latency {rep}: " + ", ".join(f"{n} {v['expand_ms']:.0f} ms" for n, v in rec["codecs"].items())
            + f"; text-mode prefill {rec['recompute_text_mode_ms']:.0f} ms")
        torch.cuda.empty_cache()

    # (3) VRAM: the whole store on the GPU, a 32K active window, one question answered from the store
    sizes = next(r for r in JsonlLog(OUT / "part4.jsonl").rows() if r["id"] == "sizes")
    window_bytes = int(cfg["part4"]["window_tokens"]) * 114688
    for n in names:
        rid = f"vram|{n}"
        if rid in done or "+" in n:
            continue
        need = sizes["codecs"][n]["store_bytes"] + sizes["codecs"][n]["shared_bytes"] + weights_bytes + window_bytes
        rec = {"id": rid, "codec": n, "store_bytes": sizes["codecs"][n]["store_bytes"], "weights_bytes": weights_bytes, "window_bytes": window_bytes,
               "arithmetic_gib": need / GIB, "vram_gib": cfg["part4"]["vram_gib"], "git_commit": env["git_commit"], "git_dirty": env["git_dirty"]}
        if need > float(cfg["part4"]["vram_gib"]) * GIB:
            rec.update({"attempted": False, "fits": False})
            out.append(rec)
            log(f"part4 vram {n}: {need / GIB:.1f} GiB by arithmetic, above {cfg['part4']['vram_gib']} GiB: not attempted")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        store = {}
        try:
            for j in range(n_pages):
                a, b = int(pb[j]), int(pb[j + 1])
                store[j] = codecs[n].encode(encode_page(model, P, ids[a:b]), start)
            stored = torch.cuda.memory_allocated()
            segs = []
            for j_i, j in enumerate(sel):
                if j_i:
                    segs.append(G)
                segs.append(codecs[n].decode(store[j]))
            cache, n_ctx = expand_kv(P, segs, ctx.inv_freq)
            del segs
            # The active window: the stitched budget, the question and the answer, in the library buffer that
            # generation allocates (as KV mode does at query time).
            gen = generate_greedy(model, cache, suffix_ids, 8, [], 2048)
            del cache
            sync()
            rec.update({"attempted": True, "fits": True, "store_allocated_gib": stored / GIB, "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB, "context_tokens": int(n_ctx), "generated": len(gen)})
        except torch.cuda.OutOfMemoryError as e:
            rec.update({"attempted": True, "fits": False, "error": str(e)[:300], "peak_allocated_gib": torch.cuda.max_memory_allocated() / GIB})
        finally:
            store.clear()
            torch.cuda.empty_cache()
        out.append(rec)
        log(f"part4 vram {n}: fits {rec['fits']}, peak allocated {rec.get('peak_allocated_gib', 0):.1f} GiB")
    log("part4: done")
    return 0


# ---------------------------------------------------------------- stage: part3 (the evaluation runner)
def part3_cells(cfg: dict) -> list[dict]:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_a6

    cfg6 = run_a6.load_config()
    ape = run_a6.ape_choice(cfg6)
    if ape is None:
        raise SystemExit("Part 3 needs results/E6_kv/ape_choice.json (A6 tuning)")
    base = run_a6.mode_config(cfg6, "kv-ape", ape=ape)
    p3 = cfg["part3"]
    names = []
    for c in p3["codecs"]:
        if c == "best_lowrank":
            names.append(choose_lowrank(cfg)["choice"])
        elif c == "int4_lossless":
            continue  # decodes to the int4 cache bit for bit: its accuracy is int4's (D-050); its cost is Part 4's
        else:
            names.append(c)
    out = []
    for name in names:
        kv = {**base["kv"], "codec": name, "group": int(cfg["group"])}
        if "lowrank" in name:
            kv["basis"] = f"{cfg['basis_dir']}/basis.pt"
        m = {**base, "kv": kv, "name": f"{base['name']}-{name}"}
        for L in p3["lengths"]:
            for t in p3["tasks"]:
                out.append({"method": m["name"], "mcfg": m, "task": t, "length": L, "n": int(p3["n"])})
    return out


def stage_part3(cfg: dict, args) -> int:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_a6
    from run_a4 import Session, cell_stats, read_rows
    from run_a5 import project

    cfg6 = run_a6.load_config()
    tcfg = load_task_config()
    todo = part3_cells(cfg)
    exp_dir = REPO_ROOT / "results" / EXP3
    run_a6._LOG_FILE = REPO_ROOT / "logs" / "A7" / "part3.log"
    run_a6._LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    session = Session()
    log(f"part3: {len(todo)} cells")
    run_a6.run_cells(todo, EXP3, cfg6, tcfg, cell_stats(read_rows(exp_dir)), session, True, ["E6_kv"])
    stats = cell_stats(read_rows(exp_dir))
    proj = project(todo, stats)
    atomic_write_json(exp_dir / "projection_part3.json", proj)
    log(f"part3: projected {proj['remaining_h']:.2f} h for what is left")
    if args.pilot:
        return 0
    if proj["remaining_h"] > args.max_hours:
        log(f"part3 is projected at {proj['remaining_h']:.1f} h, above --max-hours {args.max_hours}: rerun with a larger value")
        return 3
    summaries = run_a6.run_cells(todo, EXP3, cfg6, tcfg, stats, session, False, ["E6_kv"])
    log(f"part3 done: {sum(s['ok'] for s in summaries)} ok, {sum(s['failed'] for s in summaries)} failed")
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ("all", "part3"), required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-hours", type=float, default=12.0)
    args = ap.parse_args()
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "part3":
        return stage_part3(cfg, args)
    stages = STAGES if args.stage == "all" else (args.stage,)
    if stages == ("pages",):
        return stage_pages(cfg)
    if "pages" in stages and not pages_path(cfg).is_file():
        stage_pages(cfg)
    env = collect_env()
    atomic_write_json(OUT / "env.json", env)
    preflight(min_free_vram_gib=6.0, min_free_ram_gib=2.0, min_free_disk_gib=5.0)
    ctx = Ctx(cfg)
    try:
        with GpuTelemetry(REPO_ROOT / "logs" / "E7_codecs" / "gpu_telemetry.csv"):
            for s in stages:
                if s == "pages":
                    continue
                log(f"stage {s}")
                {"fit": stage_fit, "part1": stage_part1, "part2": stage_part2, "timing": stage_timing, "part4": stage_part4}[s](cfg, ctx)
    except Exception as e:  # noqa: BLE001
        if classify_failure(e) == "gpu_lost":
            atomic_write_json(OUT / f"interrupted_{int(time.time())}.json", interruption_record(e))
            log("GPU lost: rerun the same command once it is back")
            return EXIT_GPU_LOST
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
