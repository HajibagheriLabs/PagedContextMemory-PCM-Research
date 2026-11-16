"""Evaluation runner: rows, resume, failures, lost GPU (CPU); one tiny native run on the GPU."""
from __future__ import annotations

import json

import pytest

from pcm.eval import runner as R
from pcm.health import EXIT_GPU_LOST
from pcm.runlog import JsonlLog
from pcm.tasks.common import get_tokenizer, load_task_config
from pcm.tasks.synth import BookPool, ensure_book_cache, generate_example


@pytest.fixture(scope="module")
def cfg():
    return load_task_config()


@pytest.fixture(scope="module")
def tiny_examples(cfg):
    tok = get_tokenizer(cfg["tokenizer"])
    pool = BookPool(ensure_book_cache(tok, cfg, log=lambda m: None))
    out = []
    for task in ("S1", "S2_multikey", "S4_count"):
        for i in range(3):
            out.append({**generate_example(task, 1024, 0, i, tok, pool, length_name="1k", cfg=cfg), "benchmark": "synth"})
    return out


def _rows(tmp_path, exp="t"):
    return list(JsonlLog(tmp_path / exp / "runs.jsonl").rows())


def test_rows_and_resume(tmp_path, cfg, tiny_examples):
    mc = {"method": "echo_answer", "name": "echo"}
    s = R.run(mc, "synth:mixed", "1k", None, exp="t", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert s["ran"] == 9 and s["ok"] == 9
    rows = _rows(tmp_path)
    assert len(rows) == 9 and all(r["score"] == 1.0 and r["evidence_recall"] == 1.0 for r in rows)
    assert all(r["id"].startswith(r["method_hash"] + ":") for r in rows)
    assert (tmp_path / "t" / "env.json").is_file()
    assert json.loads((tmp_path / "t" / "methods" / f"{rows[0]['method_hash']}.json").read_text(encoding="utf-8")) == mc
    again = R.run(mc, "synth:mixed", "1k", None, exp="t", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert again["ran"] == 0 and len(_rows(tmp_path)) == 9
    other = R.run({**mc, "name": "echo2"}, "synth:mixed", "1k", None, exp="t", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert other["ran"] == 9  # a different method config is a different method


def test_pilot_runs_five_and_projects(tmp_path, cfg, tiny_examples):
    s = R.run({"method": "echo_answer", "name": "echo"}, "synth:mixed", "1k", None, exp="p", pilot=True, results_root=tmp_path,
              examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert s["ran"] == 5 and s["remaining"] == 4 and s["projected_s_for_cell"] is not None


class _Failing(R.Method):
    calls = 0

    def answer(self, example, prompt):
        _Failing.calls += 1
        raise ValueError("boom")


class _Lost(R.Method):
    def answer(self, example, prompt):
        raise RuntimeError("CUDA error: unspecified launch failure")


def test_failures_recorded_and_retried_only_on_request(tmp_path, cfg, tiny_examples, monkeypatch):
    monkeypatch.setitem(R.METHODS, "failing", _Failing)
    mc = {"method": "failing", "name": "f"}
    s = R.run(mc, "synth:mixed", "1k", 2, exp="f", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert s["failed"] == 2
    rows = _rows(tmp_path, "f")
    assert all(r["status"] == "failed" and r["failure_kind"] == "error" and "boom" in r["error"] and r["score"] is None for r in rows)
    assert R.run(mc, "synth:mixed", "1k", 2, exp="f", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)["ran"] == 0
    assert R.run(mc, "synth:mixed", "1k", 2, exp="f", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, retry_failed=True, log=lambda m: None)["ran"] == 2
    assert len(_rows(tmp_path, "f")) == 4  # append-only: the retry adds rows


def test_lost_gpu_writes_marker_and_exits_75(tmp_path, cfg, tiny_examples, monkeypatch):
    monkeypatch.setitem(R.METHODS, "lost", _Lost)
    with pytest.raises(SystemExit) as e:
        R.run({"method": "lost", "name": "l"}, "synth:mixed", "1k", 2, exp="l", results_root=tmp_path, examples=tiny_examples, allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert e.value.code == EXIT_GPU_LOST
    assert _rows(tmp_path, "l") == []
    assert list((tmp_path / "l").glob("interrupted_*.json"))


def test_dirty_tree_refused(tmp_path, cfg, tiny_examples, monkeypatch):
    import pcm.env

    real = pcm.env.collect_env
    monkeypatch.setattr(pcm.env, "collect_env", lambda: {**real(), "git_dirty": True})
    with pytest.raises(R.DirtyTree):
        R.run({"method": "echo_answer", "name": "e"}, "synth:mixed", "1k", 1, exp="d", results_root=tmp_path, examples=tiny_examples, cfg=cfg, log=lambda m: None)


def test_parse_task():
    assert R.parse_task("babilong:qa1") == ("babilong", "qa1")
    with pytest.raises(ValueError):
        R.parse_task("qa1")
    with pytest.raises(ValueError):
        R.parse_task("ruler:niah")


@pytest.mark.gpu
def test_native_tiny_run_on_gpu(request, tmp_path, cfg, tiny_examples):
    """Two S1 examples of about 1K tokens through the reader: rows carry a prediction, timings and peak VRAM."""
    from pcm.health import gpu_status

    if request.config.getoption("--no-gpu"):
        pytest.skip("GPU tests skipped on request (--no-gpu)")
    st = gpu_status()
    if not st["ok"]:
        pytest.exit(f"no usable GPU: {st['reason']}", returncode=EXIT_GPU_LOST)
    mc = {"method": "native", "name": "native-test", "reader": request.config.getoption("--reader"), "window": 32768}
    s = R.run(mc, "synth:S1", "1k", 2, exp="g", results_root=tmp_path, examples=tiny_examples[:2], allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert s["ok"] == 2, s
    rows = _rows(tmp_path, "g")
    for r in rows:
        assert isinstance(r["prediction"], str) and r["prompt_tokens"] > r["length_tokens"]
        assert r["timings"]["prefill_s"] > 0 and r["timings"]["total_s"] >= r["timings"]["prefill_s"]
        assert 0 < r["peak_vram_gib"] < 24 and r["evidence_recall"] == 1.0
    small = R.run({**mc, "name": "native-test-small", "window": 512}, "synth:S1", "1k", 1, exp="g", results_root=tmp_path,
                  examples=tiny_examples[:1], allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert small["skipped"] == 1  # longer than the method allows: skipped, not truncated
    print("native tiny run:", [(r["prediction"], r["answers"], r["score"], round(r["peak_vram_gib"], 2)) for r in rows])


def test_truncation_spans_and_even_subset():
    assert R.truncation_spans(100, 200, "tail") == [[0, 100]]
    assert R.truncation_spans(1000, 300, "tail") == [[700, 1000]]
    assert R.truncation_spans(1000, 301, "head_tail") == [[0, 150], [849, 1000]]
    with pytest.raises(ValueError):
        R.truncation_spans(10, 0, "tail")
    assert R.even_subset(590, 100)[:3] == [0, 5, 11] and len(set(R.even_subset(590, 100))) == 100
    assert R.even_subset(50, 100) == list(range(50))


def test_truncate_prepare_keeps_the_window(cfg, tiny_examples):
    """Truncation on the CPU side: prompt + answer fit the window; kept tokens are the context's own."""
    import torch

    from pcm.models import chat_segments, encode_text

    tok = get_tokenizer(cfg["tokenizer"])
    ex = tiny_examples[0]
    ctx = encode_text(tok, ex["context"])
    for mode in ("tail", "head_tail"):
        m = R.Truncate({"method": "truncate", "mode": mode, "window": 600, "reader": "reader"})
        m.prep_tokenizer = tok
        prep = m.prepare(ex, R.prompt_for(ex, cfg))
        assert len(prep["ids"]) + ex["max_new_tokens"] == 600
        kept = torch.cat([ctx[a:b] for a, b in prep["spans"]])
        n_prefix = len(encode_text(tok, chat_segments(tok, *R.prompt_for(ex, cfg), ex["context"]).prefix))
        assert torch.equal(prep["ids"][n_prefix:n_prefix + len(kept)], kept)  # the kept tokens follow the prefix unchanged
        if mode == "tail":
            assert prep["spans"][-1][1] == len(ctx)
        else:
            assert prep["spans"][0][0] == 0 and prep["spans"][-1][1] == len(ctx)
    native = R.Native({"method": "native", "window": 600, "reader": "reader"})
    native.prep_tokenizer = tok
    assert "skip" in native.prepare(ex, R.prompt_for(ex, cfg))  # whole prompt does not fit: skipped, not truncated


@pytest.mark.gpu
def test_sampling_reproducible_and_truncation_on_gpu(request, tmp_path, cfg, tiny_examples):
    from pcm.health import gpu_status

    if request.config.getoption("--no-gpu"):
        pytest.skip("GPU tests skipped on request (--no-gpu)")
    if not gpu_status()["ok"]:
        pytest.exit("no usable GPU", returncode=EXIT_GPU_LOST)
    reader = request.config.getoption("--reader")
    samp = {"method": "native", "name": "s", "reader": reader, "sampling": {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0, "seed": 0}}
    for exp in ("s1", "s2"):
        R.run(samp, "synth:S1", "1k", 2, exp=exp, results_root=tmp_path, examples=tiny_examples[:2], allow_dirty=True, cfg=cfg, log=lambda m: None)
    a, b = _rows(tmp_path, "s1"), _rows(tmp_path, "s2")
    assert [r["prediction"] for r in a] == [r["prediction"] for r in b]  # fixed seed: same samples
    tr = {"method": "truncate", "name": "t", "reader": reader, "mode": "head_tail", "window": 600}
    s = R.run(tr, "synth:S1", "1k", 2, exp="t", results_root=tmp_path, examples=tiny_examples[:2], allow_dirty=True, cfg=cfg, log=lambda m: None)
    assert s["ok"] == 2
    for r in _rows(tmp_path, "t"):
        assert r["prompt_tokens"] + 16 <= 600 and len(r["selected_spans"]) == 2


def test_answer_budget_override(cfg, tiny_examples):
    ex = {**tiny_examples[0], "benchmark": "infbench", "task": "passkey", "max_new_tokens": 6}
    m = R.Native({"method": "native", "reader": "reader", "max_new_tokens": {"infbench:passkey": 64}})
    assert m.max_new(ex) == 64
    assert m.max_new({**ex, "task": "number_string"}) == 6
    assert R.Native({"method": "native", "reader": "reader"}).max_new(ex) == 6
