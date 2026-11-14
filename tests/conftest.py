"""Shared fixtures for the engine tests.

GPU tests take the reader from the --reader option (a key of configs/assets.yaml
or a repo id; default: the development reader). Measurements that a report will
quote are written to results/<experiment>/tests_<reader>.json through `record`,
each stamped with the code commit it was measured on.

The GPU of the local machine can drop off mid-run, so:
- a missing GPU stops the session with exit code 75 instead of skipping tests
  (--no-gpu runs the CPU tests alone on purpose);
- a GPU lost during a test stops the session with exit code 75 and writes
  results/<experiment>/interrupted_<reader>.json;
- --resume skips GPU tests already recorded on the current, clean commit, so a
  rerun after a crash continues where it stopped.
"""
from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_STATE: dict = {}


def pytest_addoption(parser):
    parser.addoption("--reader", default="reader_dev", help="reader key in configs/assets.yaml or repo id")
    parser.addoption("--resume", action="store_true", help="skip GPU tests already recorded on this clean commit")
    parser.addoption("--no-gpu", action="store_true", help="run the CPU tests alone; without it a missing GPU stops the session")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and a local reader model")
    config.addinivalue_line("markers", "records(*ids): result ids this test writes through `record`")


def load_cfg() -> dict:
    with (REPO_ROOT / "configs" / "a1_engine.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_reader(key: str) -> str:
    from pcm.hub import load_assets

    return load_assets()["models"].get(key, key)


def results_dir() -> Path:
    return REPO_ROOT / "results" / load_cfg()["experiment"]


def results_path(reader_repo: str) -> Path:
    return results_dir() / f"tests_{reader_repo.split('/')[-1]}.json"


def code_stamp() -> dict:
    """Commit and cleanliness of the code under test, read once per session."""
    if "stamp" not in _STATE:
        from pcm.env import collect_env

        env = collect_env()
        _STATE["env"] = env
        _STATE["stamp"] = {"git_commit": env["git_commit"], "git_dirty": env["git_dirty"]}
    return _STATE["stamp"]


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--resume"):
        return
    stamp = code_stamp()
    if stamp["git_dirty"] or not stamp["git_commit"]:
        return  # results of uncommitted code are never reused
    path = results_path(resolve_reader(config.getoption("--reader")))
    recorded = json.loads(path.read_text(encoding="utf-8")).get("tests", {}) if path.is_file() else {}

    def is_current(test_id: str) -> bool:
        run = recorded.get(test_id, {}).get("run", {})
        return run.get("git_commit") == stamp["git_commit"] and run.get("git_dirty") is False

    for item in items:
        marker = item.get_closest_marker("records")
        if marker and marker.args and all(is_current(i) for i in marker.args):
            item.add_marker(pytest.mark.skip(reason=f"already recorded on {stamp['git_commit'][:10]} (--resume)"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    if outcome.excinfo is None or item.get_closest_marker("gpu") is None:
        return
    from pcm.health import classify_failure, interruption_record
    from pcm.runlog import atomic_write_json

    exc = outcome.excinfo[1]
    if classify_failure(exc) != "gpu_lost":
        return
    reader = resolve_reader(item.config.getoption("--reader")).split("/")[-1]
    atomic_write_json(results_dir() / f"interrupted_{reader}.json", interruption_record(exc, last_done=item.nodeid))
    _STATE["gpu_lost"] = True
    item.session.shouldstop = "GPU lost; rerun with --resume once it is back"


def pytest_sessionfinish(session, exitstatus):
    if _STATE.get("gpu_lost"):
        from pcm.health import EXIT_GPU_LOST

        session.exitstatus = EXIT_GPU_LOST


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_cfg()


@pytest.fixture(scope="session")
def reader_key(request) -> str:
    return request.config.getoption("--reader")


@pytest.fixture(scope="session")
def reader_repo(reader_key) -> str:
    return resolve_reader(reader_key)


@pytest.fixture(scope="session")
def cuda(request, cfg):
    import torch

    from pcm.health import EXIT_GPU_LOST, GpuTelemetry, gpu_status

    if request.config.getoption("--no-gpu"):
        pytest.skip("GPU tests skipped on request (--no-gpu)")
    status = gpu_status()
    if not status["ok"]:
        pytest.exit(
            f"no usable GPU: {status['reason']}. Restore the device and rerun with --resume, "
            "or pass --no-gpu to run the CPU tests alone.",
            returncode=EXIT_GPU_LOST,
        )
    telemetry = GpuTelemetry(REPO_ROOT / "logs" / cfg["experiment"] / "gpu_telemetry.csv")
    with telemetry:
        yield torch.device("cuda:0")


def free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def tokenizer(reader_repo):
    from transformers import AutoTokenizer

    from pcm.hub import model_path

    return AutoTokenizer.from_pretrained(model_path(reader_repo))


@pytest.fixture(scope="session")
def doc(cfg, tokenizer) -> dict:
    """The first PG-19 book long enough for every test: {"title", "ids" (1D LongTensor, CPU)}."""
    import torch

    from pcm.tasks.pg19 import first_long_book

    try:
        book = first_long_book(tokenizer, cfg["text"]["split"], cfg["text"]["min_tokens"])
    except LookupError as e:
        pytest.skip(str(e))
    return {"title": book["title"], "ids": torch.tensor(book["ids"], dtype=torch.long)}


@pytest.fixture(scope="session")
def doc_ids(doc):
    return doc["ids"]


@pytest.fixture(scope="session")
def record(cfg, reader_repo, doc):
    """record(test_id, dict): write one test's measurements, stamped with the code commit.

    Each call rewrites the results file atomically, so tests finished before a
    crash keep their records. Records of other tests are left as they are; the
    stamp on each one says which commit it belongs to.
    """
    from pcm.runlog import atomic_write_json

    path = results_path(reader_repo)
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    stamp = code_stamp()

    def _record(test_id: str, values: dict) -> None:
        state["reader"] = reader_repo
        state["text"] = {"source": "pg19", "split": cfg["text"]["split"], "book": doc["title"]}
        state["env"] = _STATE["env"]
        run = {**stamp, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        state.setdefault("tests", {})[test_id] = {**values, "run": run}
        atomic_write_json(path, state)

    return _record
