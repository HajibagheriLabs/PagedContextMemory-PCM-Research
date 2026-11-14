"""Shared fixtures for the engine tests.

GPU tests take the reader from the --reader option (a key of configs/assets.yaml
or a repo id; default: the development reader). Measurements that a report will
quote are written to results/<experiment>/tests_<reader>.json through `record`.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_addoption(parser):
    parser.addoption("--reader", default="reader_dev", help="reader key in configs/assets.yaml or repo id")


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and a local reader model")


@pytest.fixture(scope="session")
def cfg() -> dict:
    with (REPO_ROOT / "configs" / "a1_engine.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def reader_key(request) -> str:
    return request.config.getoption("--reader")


@pytest.fixture(scope="session")
def reader_repo(reader_key) -> str:
    from pcm.hub import load_assets

    return load_assets()["models"].get(reader_key, reader_key)


@pytest.fixture(scope="session")
def cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda:0")


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
    """record(test_id, dict): merge measurements into the results file for this reader."""
    from pcm.env import collect_env

    out_dir = REPO_ROOT / "results" / cfg["experiment"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tests_{reader_repo.split('/')[-1]}.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    state["reader"] = reader_repo
    state["text"] = {"source": "pg19", "split": cfg["text"]["split"], "book": doc["title"]}
    state["env"] = collect_env()

    def _record(test_id: str, values: dict) -> None:
        state.setdefault("tests", {})[test_id] = values
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    return _record
