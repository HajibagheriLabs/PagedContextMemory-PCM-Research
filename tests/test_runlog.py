"""Crash-safe result files and failure classification. CPU only."""
import json

import pytest
import torch

from pcm.health import EXIT_GPU_LOST, GpuLost, classify_failure, interruption_record
from pcm.runlog import JsonlLog, atomic_write_json


def test_append_and_resume(tmp_path):
    log = JsonlLog(tmp_path / "runs.jsonl")
    assert log.done_ids() == set()
    log.append({"id": "a", "correct": True})
    log.append({"id": "b", "correct": False})
    again = JsonlLog(tmp_path / "runs.jsonl")
    assert again.done_ids() == {"a", "b"}
    assert [r["id"] for r in again.rows()] == ["a", "b"]
    assert again.done_ids(where=lambda r: r["correct"]) == {"a"}


def test_truncated_last_line_is_ignored_and_not_extended(tmp_path):
    path = tmp_path / "runs.jsonl"
    log = JsonlLog(path)
    log.append({"id": "a"})
    with path.open("a", encoding="utf-8") as f:  # a crash in the middle of writing row "b"
        f.write('{"id": "b", "pred')
    resumed = JsonlLog(path)
    assert resumed.done_ids() == {"a"}
    assert resumed.bad_lines == 1
    resumed.append({"id": "b"})
    resumed.append({"id": "c"})
    assert resumed.done_ids() == {"a", "b", "c"}
    assert resumed.bad_lines == 1  # the partial line is still there, isolated
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1]) == {"id": "c"} and json.loads(lines[0]) == {"id": "a"}


def test_rows_never_span_lines(tmp_path):
    log = JsonlLog(tmp_path / "runs.jsonl")
    log.append({"id": "a", "text": "line one\nline two"})
    assert next(log.rows())["text"] == "line one\nline two"
    assert len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_atomic_write_replaces_whole_file(tmp_path):
    path = tmp_path / "state" / "ckpt.json"
    atomic_write_json(path, {"step": 1})
    atomic_write_json(path, {"step": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 2}
    assert not path.with_name("ckpt.json.tmp").exists()


@pytest.mark.parametrize(
    "exc, kind",
    [
        (RuntimeError("CUDA error: unspecified launch failure"), "gpu_lost"),
        (RuntimeError("CUDA error: an illegal memory access was encountered"), "gpu_lost"),
        (RuntimeError("GPU is lost. Reboot the system to recover this GPU"), "gpu_lost"),
        (GpuLost("nvidia-smi cannot reach the driver"), "gpu_lost"),
        (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 GiB"), "oom"),
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "oom"),
        (ValueError("bad token range"), "error"),
        (KeyError("answer"), "error"),
    ],
)
def test_classify_failure(exc, kind):
    assert classify_failure(exc) == kind


def test_interruption_record_names_the_resume_step():
    rec = interruption_record(RuntimeError("CUDA error: unspecified launch failure"), last_done="x|3")
    assert rec["kind"] == "gpu_lost" and rec["last_completed"] == "x|3"
    assert "rerun" in rec["resume"]
    assert EXIT_GPU_LOST == 75
