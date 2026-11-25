"""Evaluation runner: one method on one task cell, one durable JSON line per example.

    run(method_config, task, length, n, seed, exp=...)

task:   "synth:<S1|S2_multikey|...>", "babilong:<qa1..qa10>" or "infbench:<passkey|...>"
length: a grid name ("8k" ... "256k"; BABILong also "0k".."4k"); "full" for InfiniteBench
method_config: a dict with at least "method" (a key of METHODS) and "name".

Each example appends one row to results/<exp>/runs.jsonl (prediction, score,
timings, peak VRAM, selected token ranges, evidence recall, status, code commit).
The row id is <hash of the method config>:<example id>, so a rerun of the same
command skips what is done, and a changed method config is a different method.
Rows with status "failed" (an error or out-of-memory on that example) count as done
unless retry_failed is set; a retry appends a new row, and readers take the last
row per id. A lost GPU is never a row: the run writes interrupted_<hash>.json and
exits with status 75 (EXIT_GPU_LOST); rerun the same command once the GPU is back.

Pilot mode runs the first `pilot_n` (5) pending examples and prints seconds per
example, peak VRAM and the projected time for the whole cell.

Methods that hand the reader a selection of index units (retrieval M2, oracle M3)
add a "selection" record to the row (kept units as a bitmap, counts, unit recall
and precision, the rank of each evidence item) and a "prompt_sha". When such a
method builds a prompt that is token-for-token identical to one already answered
in the same experiment with the same reader settings, the stored answer is reused
and the row names its source in "reused_from".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from pcm.eval import metrics
from pcm.runlog import JsonlLog, atomic_write_json
from pcm.tasks.common import REPO_ROOT, config_hash, get_tokenizer, iter_jsonl, load_task_config, repo_path

ROW_SCHEMA = 1
GIB = 1024**3


class Skip(Exception):
    """The method cannot run this example (recorded as status "skipped" with the reason)."""


class DoesNotFit(Skip):
    """The prompt is longer than the method allows."""


class DirtyTree(RuntimeError):
    """Code or configuration has uncommitted changes."""


# ---------------------------------------------------------------- examples
def parse_task(task: str) -> tuple[str, str]:
    if ":" not in task:
        raise ValueError(f"task must be '<benchmark>:<name>', got {task!r}")
    bench, name = task.split(":", 1)
    if bench not in ("synth", "babilong", "infbench"):
        raise ValueError(f"unknown benchmark {bench!r}")
    return bench, name


def even_subset(total: int, n: int) -> list[int]:
    """n row indices spread evenly over 0..total-1 (all of them if n >= total)."""
    if n >= total:
        return list(range(total))
    return [(i * total) // n for i in range(n)]


def load_examples(task: str, length: str, n: int | None, seed: int, cfg: dict | None = None, split: str = "test") -> list[dict]:
    """n examples of a cell, in the harness's common form.

    Synthetic and BABILong: the first n. InfiniteBench: n rows spread evenly over the
    file (`even_subset`), so that a subset covers every needle depth; the same n
    gives the same rows to every method. split: "train" reads the synthetic train split
    (seeds >= the configured seed limit; used only for tuning).
    """
    cfg = cfg or load_task_config()
    bench, name = parse_task(task)
    tok = get_tokenizer(cfg["tokenizer"])
    if bench == "synth":
        from pcm.tasks.synth import data_file

        if split == "train" and seed < int(cfg["synth"]["seed_limit"]):
            raise ValueError(f"train seeds start at {cfg['synth']['seed_limit']}")
        path = data_file(name, length, seed, cfg, split=split)
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: materialise it with scripts/make_data.py")
        out = []
        for row in iter_jsonl(path):
            if n is not None and len(out) >= n:
                break
            out.append({**row, "benchmark": "synth"})
        if n is not None and len(out) < n:
            raise ValueError(f"{path} holds {len(out)} examples, {n} requested")
        return out
    if seed != 0 or split != "test":
        raise ValueError(f"{bench} has one fixed test set; use seed 0")
    if bench == "babilong":
        from pcm.tasks import babilong

        return babilong.load(name, length, tok, cfg, n=n)
    from pcm.tasks import infbench

    if length != "full":
        raise ValueError("InfiniteBench examples have their own lengths; use length 'full'")
    if n is None:
        return infbench.load(name, tok, cfg)
    total = len(infbench.ensure_meta(name, tok, cfg))
    return infbench.load(name, tok, cfg, select=even_subset(total, n))


def prompt_for(example: dict, cfg: dict | None = None) -> tuple[str, str, str]:
    """(system, user text before the context, user text after it) for any example."""
    bench = example.get("benchmark", "synth")
    if bench == "synth":
        from pcm.tasks.synth import prompt_parts

        return prompt_parts(example, cfg)
    if bench == "babilong":
        from pcm.tasks.babilong import prompt_parts

        return prompt_parts(example["task"], example["question"], cfg)
    from pcm.tasks.infbench import prompt_parts

    return prompt_parts(example["row"], example["task"], cfg)


# ---------------------------------------------------------------- methods
class Method:
    """A way of answering an example. Two steps, so that the runner can overlap them:
    prepare() does the CPU work (tokenizing a long context) and may run in a helper
    thread while the previous example's execute() keeps the GPU busy; execute() does
    the rest and returns {"prediction", "prompt_tokens", "selected_spans",
    "selected_pages", "timings"}. Simple methods implement answer() only.
    """

    needs_gpu = False
    reuses_answers = False  # True: run() hands the method the answers already stored for identical prompts

    def __init__(self, config: dict):
        self.config = config
        self.answer_cache: dict = {}

    def setup(self) -> None:
        pass

    def prepare(self, example: dict, prompt: tuple[str, str, str]) -> dict:
        return {"example": example, "prompt": prompt}

    def execute(self, prepared: dict) -> dict:
        return self.answer(prepared["example"], prepared["prompt"])

    def answer(self, example: dict, prompt: tuple[str, str, str]) -> dict:
        raise NotImplementedError


class EchoAnswer(Method):
    """Harness tests only: answers with the gold answer. Never a result."""

    def answer(self, example, prompt):
        t0 = time.perf_counter()
        gold = example.get("label", example["answers"])
        pred = gold[0] if example.get("benchmark") == "infbench" and example["task"] == "longbook_choice_eng" else ", ".join(map(str, example["answers"]))
        return {"prediction": pred, "prompt_tokens": None, "selected_spans": [[0, example["length_tokens"]]],
                "selected_pages": None, "timings": {"total_s": time.perf_counter() - t0}}


_READERS: dict = {}  # (reader, dtype, YaRN factor) -> (model, tokenizer, tokenizer): shared by methods on the same reader
_INDEX_MODELS: dict = {}  # ("embedder" | "reranker", key, dtype) -> model wrapper, shared by the retrieval methods


def release_readers(keep_index_models: bool = False) -> None:
    """Drop every loaded reader (and, unless told otherwise, embedder and reranker) and return the memory to the driver."""
    import gc

    _READERS.clear()
    if not keep_index_models:
        _INDEX_MODELS.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class Native(Method):
    """The reader on the whole prompt: prefill in blocks, then decoding, thinking off.

    config: reader (assets key), dtype (bfloat16), block (2048), window (32768: prompt
    plus max_new_tokens must fit, else the example is skipped, never truncated),
    max_new_tokens (optional {"bench:task": n}: answer budget replacing the task's own),
    sampling (absent: greedy; else temperature, top_k, top_p, min_p and seed; the
    generator of each example is seeded by (seed, example id)), min_free_vram_gib,
    yarn_factor (absent: the model's own RoPE).
    """

    needs_gpu = True

    def reader_key(self) -> tuple:
        factor = self.config.get("yarn_factor")
        return (self.config["reader"], self.config.get("dtype", "bfloat16"), None if factor is None else float(factor))

    def setup(self):
        import torch

        from pcm.models import load_reader, stop_ids

        key = self.reader_key()
        if key not in _READERS:
            from transformers import AutoTokenizer

            from pcm.hub import model_path

            model, tok = load_reader(key[0], yarn_factor=key[2], dtype=getattr(torch, key[1]))
            tok.model_max_length = 10**12
            # A second tokenizer for prepare(), which runs in a helper thread: one fast tokenizer
            # used from two threads at once can fail ("Already borrowed").
            prep_tok = AutoTokenizer.from_pretrained(model_path(key[0]))
            prep_tok.model_max_length = 10**12
            _READERS[key] = (model, tok, prep_tok)
        self.model, self.tokenizer, self.prep_tokenizer = _READERS[key]
        self.stops = stop_ids(self.tokenizer)

    def _segments(self, example, prompt):
        from pcm.models import chat_segments, encode_text

        tok = self.prep_tokenizer
        system, before, after = prompt
        chat = chat_segments(tok, system, before, after, example["context"])
        ctx_ids = encode_text(tok, chat.context)
        if len(ctx_ids) != example["length_tokens"]:
            raise RuntimeError(f"context tokenizes to {len(ctx_ids)} tokens, the data says {example['length_tokens']}")
        return encode_text(tok, chat.prefix), ctx_ids, encode_text(tok, chat.suffix)

    def max_new(self, example: dict) -> int:
        """The answer budget: the task's own, unless config max_new_tokens names this task ("bench:task": n)."""
        override = self.config.get("max_new_tokens") or {}
        key = f"{example.get('benchmark', 'synth')}:{example['task']}"
        return int(override.get(key, example["max_new_tokens"]))

    def window_room(self, prefix_ids, suffix_ids, max_new: int) -> int:
        """Context tokens that fit in the window beside the prompt wrapper and the answer."""
        return self.window_size() - len(prefix_ids) - len(suffix_ids) - max_new

    def prepare(self, example, prompt):
        import torch

        prefix, ctx, suffix = self._segments(example, prompt)
        max_new = self.max_new(example)
        if len(ctx) > self.window_room(prefix, suffix, max_new):
            total = len(prefix) + len(ctx) + len(suffix)
            return {"example": example, "skip": f"prompt {total} + answer {max_new} tokens exceed the window {self.window_size()}"}
        return {"example": example, "ids": torch.cat([prefix, ctx, suffix]), "spans": [[0, int(len(ctx))]]}

    def window_size(self) -> int:
        return int(self.config.get("window", 32768))

    def prompt_sha(self, example: dict, ids) -> str:
        """Hash of everything that decides a greedy answer: reader, block size, answer budget, stop ids, prompt ids."""
        import numpy as np

        h = hashlib.sha256()
        h.update(json.dumps({"reader": list(self.reader_key()), "block": int(self.config.get("block", 2048)),
                             "max_new": self.max_new(example), "stops": list(self.stops)}, sort_keys=True).encode("utf-8"))
        h.update(np.ascontiguousarray(ids.numpy(), dtype="<i8").tobytes())
        return h.hexdigest()[:24]

    def _generate(self, example: dict, ids) -> tuple[str, dict]:
        """The reader's answer to the prompt `ids`: (text, timings)."""
        from pcm.prefill import generate_greedy, generate_sampled

        max_new = self.max_new(example)
        block = int(self.config.get("block", 2048))
        timings: dict = {}
        sp = self.config.get("sampling")
        if sp:
            from pcm.tasks.common import stable_int

            out = generate_sampled(self.model, None, ids, max_new, self.stops, block, temperature=sp["temperature"],
                                   top_k=sp["top_k"], top_p=sp["top_p"], min_p=sp.get("min_p", 0.0),
                                   seed=stable_int(sp["seed"], example["id"]), timings=timings)
        else:
            out = generate_greedy(self.model, None, ids, max_new, self.stops, block, timings=timings)
        return self.tokenizer.decode(out, skip_special_tokens=True).strip(), timings

    def execute(self, prepared):
        if "skip" in prepared:
            raise DoesNotFit(prepared["skip"])
        pred, timings = self._generate(prepared["example"], prepared["ids"])
        return {"prediction": pred, "prompt_tokens": int(len(prepared["ids"])), "selected_spans": prepared["spans"],
                "selected_pages": None, "timings": timings}


def truncation_spans(n: int, keep: int, mode: str) -> list[list[int]]:
    """Token ranges kept when a context of n tokens is truncated to `keep` tokens.

    mode "tail": the last `keep` tokens; "head_tail": the first keep // 2 and the
    last keep - keep // 2. A context that fits is kept whole.
    """
    if keep <= 0:
        raise ValueError("no room for any context token")
    if n <= keep:
        return [[0, n]]
    if mode == "tail":
        return [[n - keep, n]]
    if mode == "head_tail":
        head = keep // 2
        return [[0, head], [n - (keep - head), n]]
    raise ValueError(f"unknown truncation mode {mode!r}")


class Truncate(Native):
    """M0: the reader on a truncated context that fills the window (specification, section 5.3).

    config as Native, plus mode: "tail" (the last tokens) or "head_tail" (first half
    and last half of the room). The room is window - prompt wrapper - max_new_tokens,
    so prompt and answer always fit. Cuts are made in token space; the two kept parts
    of head_tail are joined directly.
    """

    def prepare(self, example, prompt):
        import torch

        prefix, ctx, suffix = self._segments(example, prompt)
        room = self.window_room(prefix, suffix, self.max_new(example))
        spans = truncation_spans(len(ctx), room, self.config["mode"])
        kept = torch.cat([ctx[a:b] for a, b in spans])
        return {"example": example, "ids": torch.cat([prefix, kept, suffix]), "spans": spans}


class DenseYarn(Native):
    """M1: the reader with training-free YaRN on the whole prompt, dense attention (specification,
    section 5.3). config as Native, with yarn_factor: the smallest of 2, 4, 8, 16 whose extended
    range covers the longest prompt of the split (section 4.7 rule 5; set per cell by the caller).
    The window is the extended position range, not the stated context length.
    """

    def setup(self):
        from pcm.models import rope_summary

        if self.config.get("yarn_factor") is None:
            raise ValueError("dense_yarn needs yarn_factor")
        super().setup()
        rope = rope_summary(self.model)
        if rope["rope_type"] != "yarn" or float(rope["factor"]) != float(self.config["yarn_factor"]):
            raise RuntimeError(f"the loaded reader does not carry YaRN factor {self.config['yarn_factor']}: {rope}")
        self.max_positions = int(self.model.config.max_position_embeddings)

    def window_size(self) -> int:
        return self.max_positions


def query_text(example: dict) -> str:
    """What a retriever is asked: the example's question; for multiple choice, the options after it, one per line."""
    q = str(example["question"]).strip()
    options = (example.get("row") or {}).get("options") or []
    return q + ("\n" + "\n".join(str(o) for o in options) if options else "")


class UnitMethod(Native):
    """Base of the methods that give the reader a selection of index units inside the native window.

    config as Native, plus unit (128) and snap (32): index units (pcm/chunking.py); budget (28672):
    tokens of kept units; gap_marker: text put between kept spans that are not adjacent.
    Kept units are always placed in document order (text expansion, pcm/expand.py).
    """

    reuses_answers = True

    def setup(self):
        super().setup()
        self.setup_units()

    def setup_units(self) -> None:
        from pcm.models import encode_text

        self.unit, self.snap = int(self.config.get("unit", 128)), int(self.config.get("snap", 32))
        self.budget = int(self.config.get("budget", 28672))
        self.gap_ids = encode_text(self.tokenizer, self.config.get("gap_marker", "\n[...]\n"))

    def prepare(self, example, prompt):
        from pcm.chunking import cut_document
        from pcm.models import chat_segments, encode_text

        tok = self.prep_tokenizer
        system, before, after = prompt
        chat = chat_segments(tok, system, before, after, example["context"])
        units = cut_document(tok, chat.context, self.unit, self.snap)
        if units.n_tokens != example["length_tokens"]:
            raise RuntimeError(f"context tokenizes to {units.n_tokens} tokens, the data says {example['length_tokens']}")
        out = {"example": example, "units": units, "prefix": encode_text(tok, chat.prefix), "suffix": encode_text(tok, chat.suffix)}
        out.update(self.prepare_more(example, units))
        return out

    def prepare_more(self, example: dict, units) -> dict:
        return {}

    def _answer(self, prepared: dict, spans: list, kept, order, timings: dict, extra: dict | None = None) -> dict:
        """Expand `spans`, answer (or reuse the stored answer of an identical prompt) and describe the selection.

        kept: the kept unit indices (None when the selection is not made of whole units);
        order: the ranking the selection came from (None when there is none).
        """
        import numpy as np
        import torch

        from pcm.expand import expand_text
        from pcm.index import select as S

        ex, units = prepared["example"], prepared["units"]
        body, gaps = expand_text(torch.from_numpy(units.ids.astype(np.int64)), spans, self.gap_ids)
        ids = torch.cat([prepared["prefix"], body, prepared["suffix"]])
        max_new = self.max_new(ex)
        if len(ids) + max_new > self.window_size():
            raise DoesNotFit(f"prompt {len(ids)} + answer {max_new} tokens exceed the window {self.window_size()}")
        sha = self.prompt_sha(ex, ids)
        selection_only = bool(self.config.get("selection_only"))
        hit = None if (self.config.get("sampling") or selection_only) else self.answer_cache.get(sha)
        if selection_only:  # the selection is what is measured: no answer is generated
            pred = None
        elif hit is not None:
            pred = hit["prediction"]
            timings.update({"reused": True, "new_tokens": hit.get("new_tokens")})
        else:
            pred, gen = self._generate(ex, ids)
            timings.update(gen)
        items = S.evidence_units(ex.get("evidence") or [], units.bounds)
        gold = set(u for it in items for u in it)
        kept_units = set(int(u) for u in kept) if kept is not None else metrics.units_overlapping(spans, units.bounds)
        rp = metrics.unit_recall_precision(gold, kept_units)
        selection = {"n_units": units.n_units, "kept_units": len(kept_units), "kept_tokens": int(sum(b - a for a, b in spans)),
                     "gaps": gaps, "gold_units": len(gold), "unit_recall": rp["recall"], "unit_precision": rp["precision"]}
        if kept is not None:
            selection["units_b64"] = S.pack_units(kept_units, units.n_units)
        if order is not None and items:
            selection["evidence_ranks"] = S.item_ranks(order, items)
        selection.update(extra or {})
        out = {"prediction": pred, "prompt_tokens": int(len(ids)), "selected_spans": spans, "store_spans": kept is None,
               "selected_pages": None, "timings": timings, "selection": selection, "prompt_sha": sha,
               "reused_from": hit["id"] if hit is not None else None}
        if selection_only:
            out["selection_only"] = True
        return out


class Retrieve(UnitMethod):
    """M2: plain retrieval over index units, text expansion (specification, sections 4.3 and 5.3).

    config as UnitMethod, plus ranker: "bm25", "embed", "hybrid" (reciprocal-rank fusion of the two)
    or "hybrid_rerank" (the reranker re-orders the top rerank_top units of the fused ranking);
    rrf_k (60); bm25 {k1, b, epsilon}; rerank_top (64); embedder, reranker (assets keys);
    index_cache (directory of the per-document index cache). Units are added in rank order while
    they fit the budget, then put in document order.

    The CPU work (cutting, BM25, reading cached embeddings) is done in prepare(); a document whose
    embeddings are not cached yet is embedded in execute() and stored ("ingest_s" in the timings).
    """

    RANKERS = ("bm25", "embed", "hybrid", "hybrid_rerank")

    def setup(self):
        self.setup_ranker()
        super().setup()
        self.setup_index_models()

    def setup_ranker(self) -> None:
        self.ranker = self.config["ranker"]
        if self.ranker not in self.RANKERS:
            raise ValueError(f"unknown ranker {self.ranker!r}; known: {self.RANKERS}")
        self.uses_bm25 = self.ranker != "embed"
        self.uses_emb = self.ranker != "bm25"
        self.embedder = self.reranker = self.cache = None

    def setup_index_models(self) -> None:
        """Embedder (with the per-document index cache) and reranker, as the ranker needs them."""
        import torch

        dtype_name = self.config.get("dtype", "bfloat16")
        if self.uses_emb:
            key = ("embedder", self.config.get("embedder", "embedder"), dtype_name)
            if key not in _INDEX_MODELS:
                from pcm.index.embed import Embedder

                _INDEX_MODELS[key] = Embedder(key[1], dtype=getattr(torch, dtype_name), batch_size=int(self.config.get("embed_batch", 128)))
            self.embedder = _INDEX_MODELS[key]
            self.open_cache()
        if self.ranker == "hybrid_rerank":
            key = ("reranker", self.config.get("reranker", "reranker"), dtype_name)
            if key not in _INDEX_MODELS:
                from pcm.index.rerank import Reranker

                _INDEX_MODELS[key] = Reranker(key[1], dtype=getattr(torch, dtype_name), batch_size=int(self.config.get("rerank_batch", 32)))
            self.reranker = _INDEX_MODELS[key]

    def open_cache(self) -> None:
        from pcm.index.ingest import IndexCache
        from pcm.tasks.common import repo_path, tokenizer_tag

        self.cache = IndexCache(repo_path(self.config.get("index_cache", "data/cache/index")), tokenizer_tag(self.tokenizer),
                                self.embedder.tag, self.unit, self.snap)

    def sync(self) -> None:
        """Wait for the GPU, so that a timing covers the work it names."""
        if self.needs_gpu:
            from pcm.prefill import _sync

            _sync(self.model.device)

    def prepare_more(self, example, units):
        from pcm.index.bm25 import BM25
        from pcm.index.ingest import doc_hash

        texts = units.texts(example["context"])
        query = query_text(example)
        out = {"texts": texts, "query": query}
        if self.uses_bm25:
            t = time.perf_counter()
            b = self.config.get("bm25") or {}
            out["bm25"] = BM25.from_texts(texts, k1=b.get("k1", 1.5), b=b.get("b", 0.75), epsilon=b.get("epsilon", 0.25)).scores(query)
            out["bm25_s"] = time.perf_counter() - t
        if self.uses_emb:
            out["doc"] = doc_hash(units.ids)
            out["emb"] = self.cache.load(out["doc"], units.bounds)
        return out

    def execute(self, prepared):
        import numpy as np

        from pcm.index import select as S
        from pcm.index.ingest import ingest_document

        ex, units = prepared["example"], prepared["units"]
        timings: dict = {}
        extra = {"ranker": self.ranker}
        if self.uses_emb:
            emb = prepared["emb"]
            if emb is None:
                t = time.perf_counter()
                emb, _ = ingest_document(units, ex["context"], self.embedder, self.cache, {"example": ex["id"]})
                timings["ingest_s"] = time.perf_counter() - t
        t0 = time.perf_counter()
        if self.uses_emb:
            cos = emb @ self.embedder.encode_query(prepared["query"])
        if self.ranker == "bm25":
            order = S.order_by_score(prepared["bm25"])
        elif self.ranker == "embed":
            order = S.order_by_score(cos)
        else:
            order = S.order_by_score(S.rrf([cos, prepared["bm25"]], float(self.config.get("rrf_k", 60))))
        if self.ranker == "hybrid_rerank":
            t = time.perf_counter()
            top = order[: int(self.config.get("rerank_top", 64))]
            scores = self.reranker.score(prepared["query"], [prepared["texts"][int(u)] for u in top])
            self.sync()
            head = S.rerank_head(order, scores)
            extra["rerank_moved"] = int(np.sum(head[: len(top)] != top))  # top units whose place the reranker changed
            order = head
            timings["rerank_s"] = time.perf_counter() - t
        kept = S.fill_budget(order, units.sizes, self.budget)
        spans = S.spans_of(kept, units.bounds)
        self.sync()
        timings["select_s"] = time.perf_counter() - t0 + prepared.get("bm25_s", 0.0)
        return self._answer(prepared, spans, kept, order, timings, extra)


class PCMText(Retrieve):
    """PCM, text mode (specification, sections 4.3 to 4.5; D-041 to D-044): hybrid scores over index units,
    optional graph expansion, units mapped to pages, the budget's nominal number of pages kept, document
    order, text expansion with gap markers.

    config as Retrieve (ranker must be "hybrid"), plus page (tokens, a multiple of unit), budget (tokens),
    graph (absent or null: none; else {traversal: "ppr" with edges (any of seq, sem, ent), or "hops" with
    hops (1 or 2); seeds (16), restart (0.5), iterations (10), sem_k (5), sem_min_cos (0.5), ent_cap (20),
    hop_k (16), extractor ("default" or "spacy")}), selection_only (true: record the selection and its
    evidence recall, generate nothing; no reader is loaded).
    """

    def setup(self):
        self.check_config()
        self.setup_ranker()
        if self.config.get("selection_only"):
            self._setup_tokenizer_only()
        else:
            Native.setup(self)
        self.setup_units()
        self.setup_index_models()
        self.setup_pages()

    def check_config(self) -> None:
        if self.config.get("ranker", "hybrid") != "hybrid":
            raise ValueError("PCM scores units by hybrid retrieval (section 4.3, query step 1)")
        g = self.config.get("graph") or {}
        self.graph_cfg = g
        if g:
            from pcm.index.graph import EDGE_TYPES, EXTRACTORS

            if g.get("traversal") not in ("ppr", "hops"):
                raise ValueError(f"graph traversal must be 'ppr' or 'hops', not {g.get('traversal')!r}")
            if g["traversal"] == "ppr" and (not g.get("edges") or any(e not in EDGE_TYPES for e in g["edges"])):
                raise ValueError(f"PPR needs edges from {EDGE_TYPES}")
            if g["traversal"] == "hops" and int(g.get("hops", 0)) < 1:
                raise ValueError("entity hops need hops >= 1")
            if g.get("extractor", "default") not in EXTRACTORS:
                raise ValueError(f"unknown entity extractor {g.get('extractor')!r}")

    def setup_pages(self) -> None:
        from pcm.chunking import units_per_page
        from pcm.index.graph import GraphCache

        self.page = int(self.config["page"])
        self.per_page = units_per_page(self.page, self.unit)
        if self.page > self.budget:
            raise ValueError(f"a page of {self.page} tokens does not fit the budget of {self.budget}")
        # Beside the document's embeddings: same tokenizer, embedder and unit cut.
        self.graph_cache = GraphCache(self.cache.dir)

    def reader_key(self) -> tuple:
        if self.config.get("selection_only"):
            return ("selection-only", self.config["reader"])
        return super().reader_key()

    def _setup_tokenizer_only(self) -> None:
        from transformers import AutoTokenizer

        from pcm.hub import model_path
        from pcm.models import stop_ids

        path = model_path(self.config["reader"])
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.prep_tokenizer = AutoTokenizer.from_pretrained(path)
        for t in (self.tokenizer, self.prep_tokenizer):
            t.model_max_length = 10**12
        self.stops = stop_ids(self.tokenizer)

    def sync(self) -> None:
        if self.model is not None:
            return super().sync()
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _needs(self) -> tuple[bool, bool]:
        """(entities, SEM edges) needed by the graph configuration."""
        g = self.graph_cfg
        if not g:
            return False, False
        if g["traversal"] == "hops":
            return True, False
        return "ent" in g["edges"], "sem" in g["edges"]

    def prepare_more(self, example, units):
        out = super().prepare_more(example, units)
        need_ent, _ = self._needs()
        if need_ent:
            from pcm.index import graph as G

            g = self.graph_cfg
            extractor, cap = g.get("extractor", "default"), int(g.get("ent_cap", 20))
            name = G.ent_name(extractor, cap)
            z = self.graph_cache.load(name, out["doc"])
            if z is None:
                t = time.perf_counter()
                ents = G.extract(out["texts"], extractor)
                edges = G.ent_edges(ents, cap)
                size = self.graph_cache.save(name, out["doc"], G.pack_entities(ents, edges))
                out["ent_build_s"] = time.perf_counter() - t
                out["ent_bytes"] = size
            else:
                ents, edges = G.unpack_entities(z)
            out["ents"], out["ent_edges"] = ents, edges
            t = time.perf_counter()
            out["q_ents"] = G.question_entities(out["query"], ents, extractor)
            out["q_ent_s"] = time.perf_counter() - t
        return out

    def execute(self, prepared):
        s = self.select(prepared)
        return self._answer(prepared, s["spans"], s["kept"], s["order"], s["timings"], s["extra"])

    def select(self, prepared) -> dict:
        """Hybrid scores, graph, pages, budget: {spans, kept (units), kept_pages, order, timings, extra}."""
        import numpy as np

        from pcm.index import graph as G
        from pcm.index import select as S
        from pcm.index.ingest import ingest_document

        ex, units = prepared["example"], prepared["units"]
        timings: dict = {}
        extra: dict = {"ranker": "hybrid", "page": self.page, "budget": self.budget}
        emb = prepared["emb"]
        if emb is None:
            t = time.perf_counter()
            emb, _ = ingest_document(units, ex["context"], self.embedder, self.cache, {"example": ex["id"]})
            timings["ingest_s"] = time.perf_counter() - t
        for k in ("ent_build_s",):
            if k in prepared:
                timings[k] = prepared[k]
        g = self.graph_cfg
        need_ent, need_sem = self._needs()
        sem = None
        if need_sem:
            name = G.sem_name(int(g.get("sem_k", 5)), float(g.get("sem_min_cos", 0.5)))
            z = self.graph_cache.load(name, prepared["doc"])
            if z is None:
                t = time.perf_counter()
                sem = G.sem_edges(emb, int(g.get("sem_k", 5)), float(g.get("sem_min_cos", 0.5)))
                self.sync()
                self.graph_cache.save(name, prepared["doc"], {"rows": sem[0], "cols": sem[1], "weights": sem[2]})
                timings["sem_build_s"] = time.perf_counter() - t
            else:
                sem = (z["rows"], z["cols"], z["weights"])

        t0 = time.perf_counter()
        k_rrf = float(self.config.get("rrf_k", 60))
        cos = emb @ self.embedder.encode_query(prepared["query"])
        fused = S.rrf([cos, prepared["bm25"]], k_rrf)
        base = S.order_by_score(fused)
        order = base
        if g:
            seeds = int(g.get("seeds", 16))
            t = time.perf_counter()
            if g["traversal"] == "ppr":
                edges = {}
                if need_sem:
                    edges["sem"] = sem
                if need_ent:
                    edges["ent"] = prepared["ent_edges"]
                trans = G.Graph(units.n_units, edges).transition(g["edges"])
                scores = G.ppr(trans, base[:seeds], float(g.get("restart", 0.5)), int(g.get("iterations", 10)))
                order = S.order_by_score(S.rrf([fused, scores], k_rrf))
                extra["graph"] = {"traversal": "ppr", "edges": list(g["edges"]), "edges_out": int(trans.nnz),
                                  "reached": int(np.sum(scores > 0))}
            else:
                order, added = G.entity_hops(base, prepared["ents"], prepared["q_ents"], int(g["hops"]), seeds, int(g.get("hop_k", 16)))
                extra["graph"] = {"traversal": "hops", "hops": int(g["hops"]), "added": added, "question_entities": len(prepared["q_ents"])}
            self.sync()
            timings["graph_s"] = time.perf_counter() - t
            extra["graph"]["extractor"] = g.get("extractor", "default") if need_ent else None
            extra["graph"]["moved_in_top_seeds"] = int(np.sum(order[:seeds] != base[:seeds]))
        pages = S.page_order(order, self.per_page)
        kept_pages = S.fill_pages(pages, self.page, self.budget)
        kept = S.units_of_pages(kept_pages, self.per_page, units.n_units)
        spans = S.spans_of(kept, units.bounds)
        self.sync()
        timings["select_s"] = time.perf_counter() - t0 + prepared.get("bm25_s", 0.0) + prepared.get("q_ent_s", 0.0)
        items = S.evidence_units(ex.get("evidence") or [], units.bounds)
        extra.update({"n_pages": int(-(-units.n_units // self.per_page)), "kept_pages": int(len(kept_pages))})
        if items:
            extra["evidence_page_ranks"] = S.item_page_ranks(pages, items, self.per_page)
        return {"spans": spans, "kept": kept, "kept_pages": kept_pages, "order": order, "timings": timings, "extra": extra}


class PCMKV(PCMText):
    """PCM, KV mode (specification, section 4.5; D-046 to D-048): the page selection of text mode, the same
    token sequence in the window, but built from caches: the shared prefix P (system, instruction, everything
    before the context) is encoded once per prompt template; every selected page, and the gap marker, is
    encoded as a continuation of P (attending to P and to itself), kept in a page store (GPU while it fits,
    then disk), and stitched after P in document order with its keys re-based; only the question is prefilled
    at query time.

    config as PCMText, plus kv: {temperature (1.0), scale (1.0): APE-style fix, keys of context tokens divided
    by the temperature and log(scale) added to the question's attention scores on context columns;
    store_gpu_gib (8.0): page bytes kept on the GPU before the store spills to disk}.
    """

    def setup(self):
        if self.config.get("selection_only"):
            raise ValueError("KV mode generates answers; selection-only rows belong to text mode")
        super().setup()
        from pcm.store.kv import rope_inv_freq

        kv = self.config.get("kv") or {}
        self.temperature = float(kv.get("temperature", 1.0))
        self.scale = float(kv.get("scale", 1.0))
        if self.temperature <= 0 or self.scale <= 0:
            raise ValueError("temperature and scale must be positive")
        self.store_gpu_bytes = int(float(kv.get("store_gpu_gib", 8.0)) * 1024**3)
        self.inv_freq = rope_inv_freq(self.model)
        self._prefix_memo: dict = {}

    def _shared_caches(self, prefix_ids, timings: dict):
        """(cache of P, cache of the gap marker encoded after P), once per distinct prefix."""
        from pcm.prefill import prefill_in_blocks
        from pcm.store.kv import encode_page

        key = hashlib.sha256(prefix_ids.numpy().astype("<i8").tobytes()).hexdigest()
        if key not in self._prefix_memo:
            t = time.perf_counter()
            block = int(self.config.get("block", 2048))
            p = prefill_in_blocks(self.model, prefix_ids, block=block)
            g = encode_page(self.model, p, self.gap_ids, block=block) if len(self.gap_ids) else None
            self.sync()
            timings["prefix_encode_s"] = time.perf_counter() - t
            self._prefix_memo = {key: (p, g)}  # one template at a time: a cell has one
        return self._prefix_memo[key]

    def kv_sha(self, text_sha: str) -> str:
        """The prompt hash of text mode, extended by everything that makes the KV answer differ from it."""
        spec = {"mode": "kv", "page": self.page, "temperature": self.temperature, "scale": self.scale}
        return hashlib.sha256((text_sha + json.dumps(spec, sort_keys=True)).encode("utf-8")).hexdigest()[:24]

    def execute(self, prepared):
        import math

        import numpy as np
        import torch

        from pcm.chunking import page_bounds
        from pcm.expand import expand_kv, expand_text, kv_layout
        from pcm.prefill import generate_greedy
        from pcm.store.kv import encode_page
        from pcm.store.pages import PageStore

        s = self.select(prepared)
        ex, units = prepared["example"], prepared["units"]
        timings = s["timings"]
        ctx = torch.from_numpy(units.ids.astype(np.int64))
        body, gaps = expand_text(ctx, s["spans"], self.gap_ids)
        prefix, suffix = prepared["prefix"], prepared["suffix"]
        max_new = self.max_new(ex)
        n_text = len(prefix) + len(body) + len(suffix)
        if n_text + max_new > self.window_size():
            raise DoesNotFit(f"prompt {n_text} + answer {max_new} tokens exceed the window {self.window_size()}")
        text_sha = self.prompt_sha(ex, torch.cat([prefix, body, suffix]))
        sha = self.kv_sha(text_sha)
        pb = page_bounds(units.bounds, self.per_page)
        pages = sorted(int(p) for p in s["kept_pages"])
        layout = kv_layout([(int(pb[p]), int(pb[p + 1])) for p in pages])
        hit = self.answer_cache.get(sha)
        if hit is not None:
            pred = hit["prediction"]
            timings.update({"reused": True, "new_tokens": hit.get("new_tokens")})
        else:
            P, G = self._shared_caches(prefix, timings)
            store = PageStore(self.store_gpu_bytes, device=self.model.device)
            try:
                t = time.perf_counter()
                block = int(self.config.get("block", 2048))
                for kind, a, b in layout:  # ingest: every selected page encoded on its own after P
                    if kind == "page":
                        store.put(a, encode_page(self.model, P, ctx[a:b], block=block))
                self.sync()
                timings["kv_encode_s"] = time.perf_counter() - t
                t = time.perf_counter()  # query time: read the pages back, re-base, stitch
                segs = [store.get(a) if kind == "page" else G for kind, a, _ in layout]
                cache, n_ctx = expand_kv(P, segs, self.inv_freq, self.temperature)
                self.sync()
                timings["expand_s"] = time.perf_counter() - t
                if n_ctx != len(body):
                    raise RuntimeError(f"stitched context holds {n_ctx} tokens, the text-mode context {len(body)}")
                bias = (len(prefix), len(prefix) + n_ctx, math.log(self.scale)) if self.scale != 1.0 else None
                gen: dict = {}
                out = generate_greedy(self.model, cache, suffix, max_new, self.stops, block, timings=gen, context_bias=bias)
                timings.update(gen)
                timings.update({"store_gpu_bytes": store.bytes_gpu, "store_disk_bytes": store.bytes_disk})
                pred = self.tokenizer.decode(out, skip_special_tokens=True).strip()
                del cache, segs
            finally:
                store.clear()
        n_pages = sum(1 for kind, _, _ in layout if kind == "page")
        timings.update({"query_prefill_tokens": int(len(suffix)), "encoded_tokens": int(len(body) - gaps * len(self.gap_ids))})
        result = self._answer_record(prepared, s, gaps, text_sha, sha, pred, hit)
        result["selection"].update({"mode": "kv", "kv_pages": n_pages, "kv_segments": len(layout), "stitched_boundaries": len(layout) - 1,
                                    "temperature": self.temperature, "scale": self.scale})
        result["prompt_tokens"] = int(n_text)
        return result

    def _answer_record(self, prepared, s, gaps, text_sha, sha, pred, hit) -> dict:
        """The row fields _answer writes, for an answer produced in KV mode."""
        from pcm.index import select as S

        ex, units = prepared["example"], prepared["units"]
        items = S.evidence_units(ex.get("evidence") or [], units.bounds)
        gold = set(u for it in items for u in it)
        kept_units = set(int(u) for u in s["kept"])
        rp = metrics.unit_recall_precision(gold, kept_units)
        selection = {"n_units": units.n_units, "kept_units": len(kept_units), "kept_tokens": int(sum(b - a for a, b in s["spans"])),
                     "gaps": gaps, "gold_units": len(gold), "unit_recall": rp["recall"], "unit_precision": rp["precision"],
                     "units_b64": S.pack_units(kept_units, units.n_units), "text_prompt_sha": text_sha}
        if items:
            selection["evidence_ranks"] = S.item_ranks(s["order"], items)
        selection.update(s["extra"])
        return {"prediction": pred, "selected_spans": s["spans"], "store_spans": False, "selected_pages": None,
                "timings": s["timings"], "selection": selection, "prompt_sha": sha,
                "reused_from": hit["id"] if hit is not None else None}


class Oracle(UnitMethod):
    """M3: oracle pages (specification, section 5.3): the index units that hold gold evidence, padded with
    their neighbours, nearest first, while they fit the budget.

    config as UnitMethod, plus pad_units (absent: pad up to the budget; k: only neighbours within
    k units of a gold unit). If the gold units alone exceed the budget, every evidence item is
    kept with the same token margin on both sides, the largest that fits. An example without
    evidence spans is skipped.
    """

    def execute(self, prepared):
        import numpy as np

        from pcm.index import select as S

        ex, units = prepared["example"], prepared["units"]
        evidence = ex.get("evidence") or []
        if not evidence:
            raise Skip("the example has no evidence span: no oracle")
        t0 = time.perf_counter()
        gold = sorted(set(u for it in S.evidence_units(evidence, units.bounds) for u in it))
        pad = self.config.get("pad_units")
        if int(units.sizes[np.asarray(gold)].sum()) <= self.budget:
            order = S.oracle_order(gold, units.n_units, None if pad is None else int(pad))
            kept = S.fill_budget(order, units.sizes, self.budget)
            spans = S.spans_of(kept, units.bounds)
            extra = {"oracle": "units"}
        else:
            spans, margin = S.token_windows(evidence, units.n_tokens, self.budget)
            kept = None
            extra = {"oracle": "token_windows", "margin": margin}
        timings = {"select_s": time.perf_counter() - t0}
        return self._answer(prepared, spans, kept, None, timings, extra)


METHODS = {"native": Native, "truncate": Truncate, "dense_yarn": DenseYarn, "retrieve": Retrieve, "oracle": Oracle,
           "pcm": PCMText, "pcm_kv": PCMKV, "echo_answer": EchoAnswer}


def make_method(config: dict) -> Method:
    if config.get("method") not in METHODS:
        raise ValueError(f"unknown method {config.get('method')!r}; known: {sorted(METHODS)}")
    return METHODS[config["method"]](config)


# ---------------------------------------------------------------- run
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_example_fields(ex: dict) -> dict:
    return {
        "example_id": ex["id"],
        "benchmark": ex.get("benchmark", "synth"),
        "family": ex.get("family"),
        "index": ex.get("index"),
        "length_tokens": ex.get("length_tokens"),
        "depth_decile": ex.get("depth_decile"),
        "question": ex.get("question"),
        "answers": ex.get("answers"),
        "exclude": ex.get("exclude"),
        "label": ex.get("label"),
    }


def run(
    method_config: dict,
    task: str,
    length: str,
    n: int | None,
    seed: int = 0,
    exp: str = "scratch",
    pilot: bool = False,
    pilot_n: int = 5,
    retry_failed: bool = False,
    allow_dirty: bool = False,
    results_root: Path | None = None,
    examples: list[dict] | None = None,
    indices: list[int] | None = None,
    method: Method | None = None,
    cfg: dict | None = None,
    log=print,
    answer_from: list[str] | None = None,
) -> dict:
    """Run one method on one cell. Returns a summary (with the pilot projection in pilot mode).

    indices: keep only the examples with these index values (a fixed subset of the cell).
    method: an instance of the method already set up (shared across cells, so a reader
    is loaded once); otherwise one is built from method_config and set up here.
    answer_from: other experiments whose stored answers may be reused for identical prompts
    (the prompt hash covers reader, block size, answer budget, stop ids and every prompt token).
    """
    from concurrent.futures import ThreadPoolExecutor

    from pcm.env import collect_env
    from pcm.health import EXIT_GPU_LOST, GpuTelemetry, classify_failure, interruption_record, preflight

    cfg = cfg or load_task_config()
    exp_dir = Path(results_root or REPO_ROOT / "results") / exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    if env["git_dirty"] and not allow_dirty:
        raise DirtyTree("uncommitted code or configuration: commit first so every row carries a clean commit (or pass allow_dirty)")
    atomic_write_json(exp_dir / "env.json", env)
    JsonlLog(exp_dir / "env_history.jsonl", id_field="timestamp_utc").append({**env, "task": task, "length": length})

    mhash = config_hash(method_config)
    atomic_write_json(exp_dir / "methods" / f"{mhash}.json", method_config)
    if examples is None:
        examples = load_examples(task, length, (max(indices) + 1 if indices else n), seed, cfg)
    elif n is not None and not indices:
        examples = examples[:n]
    if indices:
        wanted = set(indices)
        examples = [e for e in examples if e.get("index") in wanted]
    runs = JsonlLog(exp_dir / "runs.jsonl")

    def counts_as_done(row: dict) -> bool:
        return row.get("status") in ("ok", "skipped") or (row.get("status") == "failed" and not retry_failed)

    done = runs.done_ids(where=counts_as_done)
    pending = [e for e in examples if f"{mhash}:{e['id']}" not in done]
    todo = pending[:pilot_n] if pilot else pending
    log(f"[{exp}] {method_config.get('name')} ({mhash}) {task} {length} seed {seed}: "
        f"{len(examples)} examples, {len(examples) - len(pending)} done, running {len(todo)}{' (pilot)' if pilot else ''}")

    if method is None:
        method = make_method(method_config)
        if todo:
            method.setup()
    answers: dict = {}
    if method.reuses_answers and todo:
        # Answers already stored in this experiment (and in `answer_from`), by prompt hash: an identical prompt is
        # not answered twice. Rows that only recorded a selection hold no answer.
        sources = [runs] + [JsonlLog(Path(results_root or REPO_ROOT / "results") / e / "runs.jsonl") for e in (answer_from or [])]
        for src in sources:
            for r in src.rows():
                if r.get("status") == "ok" and r.get("prompt_sha") and not r.get("reused_from") and not r.get("selection_only")                         and r.get("prediction") is not None:
                    answers.setdefault(r["prompt_sha"], {"id": r["id"], "prediction": r["prediction"],
                                                          "new_tokens": (r.get("timings") or {}).get("new_tokens")})
    method.answer_cache = answers
    gpu = method.needs_gpu
    if gpu:
        import torch

        if todo:
            pf = preflight(min_free_vram_gib=float(method_config.get("min_free_vram_gib", 0.0)),
                           min_free_ram_gib=float(method_config.get("min_free_ram_gib", 2.0)),
                           min_free_disk_gib=1.0, disk_path=exp_dir)
            log(f"preflight: free VRAM {pf['free_vram_gib']:.1f} GiB, RAM {pf['free_ram_gib']:.1f} GiB, disk {pf['free_disk_gib']:.0f} GiB")
    telemetry = GpuTelemetry(REPO_ROOT / "logs" / exp / "gpu_telemetry.csv") if gpu and todo else nullcontext()

    def prepare(ex: dict):
        """CPU part of one example; runs in a helper thread while the GPU works on the previous one."""
        t = time.perf_counter()
        try:
            return method.prepare(ex, prompt_for(ex, cfg)), None, time.perf_counter() - t
        except Exception as e:  # noqa: BLE001 - raised again in the main loop, where it is classified
            return None, e, time.perf_counter() - t

    walls: list[float] = []
    peaks: list[float] = []
    status_counts = {"ok": 0, "failed": 0, "skipped": 0}
    last_done = None
    with telemetry, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(prepare, todo[0]) if todo else None
        t_prev = time.perf_counter()
        for i, ex in enumerate(todo):
            prepared, prep_error, prep_s = future.result()
            future = pool.submit(prepare, todo[i + 1]) if i + 1 < len(todo) else None
            row = {"id": f"{mhash}:{ex['id']}", "schema": ROW_SCHEMA, "exp": exp, "method": method_config.get("name"),
                   "method_hash": mhash, "task": task, "length": length, "seed": seed, **_row_example_fields(ex),
                   "pilot": pilot, "git_commit": env["git_commit"], "git_dirty": env["git_dirty"]}
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            try:
                if prep_error is not None:
                    raise prep_error
                out = method.execute(prepared)
                total = time.perf_counter() - t0
                selection_only = bool(out.get("selection_only"))
                sc = None if selection_only else metrics.score(out["prediction"], ex)
                row.update({
                    "status": "ok", "prediction": out["prediction"], "score": sc, "correct": None if sc is None else sc == 1.0,
                    "prompt_tokens": out.get("prompt_tokens"), "selected_pages": out.get("selected_pages"),
                    "selected_spans": out.get("selected_spans") if out.get("store_spans", True) else None,
                    "evidence_recall": metrics.evidence_item_recall(ex.get("evidence") or [], out.get("selected_spans") or []),
                    "timings": {**out.get("timings", {}), "total_s": total, "prepare_s": prep_s},
                })
                for key in ("selection", "prompt_sha", "reused_from", "selection_only"):
                    if key in out:
                        row[key] = out[key]
                if out.get("prompt_sha") and not out.get("reused_from") and not selection_only:
                    answers.setdefault(out["prompt_sha"], {"id": row["id"], "prediction": out["prediction"],
                                                            "new_tokens": out.get("timings", {}).get("new_tokens")})
            except Skip as e:
                row.update({"status": "skipped", "reason": str(e), "score": None})
            except Exception as e:  # noqa: BLE001 - every failure is classified and recorded
                kind = classify_failure(e)
                if kind == "gpu_lost":
                    atomic_write_json(exp_dir / f"interrupted_{mhash}.json", {**interruption_record(e, last_done), "task": task, "length": length})
                    log(f"GPU lost during {ex['id']}: {e}. Nothing was recorded for it. Rerun the same command once the GPU is back.")
                    raise SystemExit(EXIT_GPU_LOST)
                row.update({"status": "failed", "failure_kind": kind, "error": f"{type(e).__name__}: {str(e)[:500]}", "score": None})
                if kind == "oom" and gpu:
                    torch.cuda.empty_cache()
            if gpu:
                row["peak_vram_gib"] = torch.cuda.max_memory_allocated() / GIB
                row["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
                peaks.append(row["peak_vram_gib"])
            row["timestamp_utc"] = _now()
            runs.append(row)
            now = time.perf_counter()
            if row["status"] == "ok":
                walls.append(now - t_prev)  # wall time per example, overlap included: what a projection needs
            t_prev = now
            status_counts[row["status"]] += 1
            last_done = ex["id"]

    remaining = len(pending) - len(todo)
    sec = sum(walls) / len(walls) if walls else None
    summary = {"exp": exp, "method": method_config.get("name"), "method_hash": mhash, "task": task, "length": length,
               "seed": seed, "examples": len(examples), "ran": len(todo), **status_counts, "remaining": remaining,
               "seconds_per_example": sec, "peak_vram_gib": max(peaks) if peaks else None}
    if pilot:
        # Skipped examples cost almost nothing: project only the share that runs.
        share = status_counts["ok"] / len(todo) if todo else 0.0
        summary["projected_s_for_cell"] = sec * len(pending) * share if sec is not None else 0.0
        summary["projected_s_remaining"] = sec * remaining * share if sec is not None else 0.0
        if sec is not None:
            log(f"pilot: {len(todo)} examples, {sec:.2f} s/example, peak VRAM {summary['peak_vram_gib'] or 0:.2f} GiB; "
                f"projected {summary['projected_s_for_cell'] / 3600:.2f} h for the {len(pending)} pending examples of this cell")
        else:
            log(f"pilot: no example ran ({status_counts})")
    return summary


def _load_method_config(spec: str) -> dict:
    p = Path(spec)
    if p.suffix in (".yaml", ".yml", ".json") and repo_path(p).is_file():
        import yaml

        return yaml.safe_load(repo_path(p).read_text(encoding="utf-8"))
    return json.loads(spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one method on one task cell; resumable.")
    ap.add_argument("--method", required=True, help="method config: a YAML/JSON file or inline JSON")
    ap.add_argument("--task", required=True)
    ap.add_argument("--length", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    summary = run(_load_method_config(args.method), args.task, args.length, args.n, args.seed, args.exp,
                  pilot=args.pilot, retry_failed=args.retry_failed, allow_dirty=args.allow_dirty)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
