"""Local-first access to Hugging Face models and datasets.

Every asset is resolved from the local hub cache first. The network is used
only when a snapshot is missing and downloads are allowed. Callers get a plain
directory path, so later loads never need the network.
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_YAML = REPO_ROOT / "configs" / "assets.yaml"


def load_assets(path: Path = ASSETS_YAML) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hub_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def resolve_snapshot(
    repo_id: str,
    repo_type: str = "model",
    allow_patterns: list[str] | None = None,
    allow_download: bool = True,
) -> tuple[Path, str]:
    """Return (snapshot directory, source) with source in {"cache", "download"}.

    A cached snapshot is not checked for completeness here; use missing_files().
    """
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(
            repo_id,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        return Path(path), "cache"
    except Exception:
        if not allow_download:
            raise
    path = snapshot_download(repo_id, repo_type=repo_type, allow_patterns=allow_patterns)
    return Path(path), "download"


def missing_files(snapshot: Path, required: list[str]) -> list[str]:
    """Relative paths from `required` that are absent or empty in the snapshot."""
    out = []
    for rel in required:
        p = snapshot / rel
        if not p.is_file() or p.stat().st_size == 0:
            out.append(rel)
    return out


def snapshot_size(snapshot: Path) -> tuple[int, int]:
    """(file count, total bytes), following the cache's links to the real blobs."""
    n = 0
    total = 0
    for p in snapshot.rglob("*"):
        if p.is_file():
            n += 1
            total += p.stat().st_size
    return n, total


def model_path(key_or_id: str, allow_download: bool = False) -> Path:
    """Snapshot directory of a model named by its key in configs/assets.yaml or by repo id."""
    models = load_assets()["models"]
    repo_id = models.get(key_or_id, key_or_id)
    path, _ = resolve_snapshot(repo_id, "model", allow_download=allow_download)
    return path


def dataset_path(
    repo_id: str,
    allow_patterns: list[str] | None = None,
    allow_download: bool = False,
) -> Path:
    """Snapshot directory of a dataset.

    Pass allow_patterns when only part of the repository is wanted: the hub
    library rejects a cached snapshot that lacks files outside the patterns.
    """
    path, _ = resolve_snapshot(repo_id, "dataset", allow_patterns, allow_download)
    return path
