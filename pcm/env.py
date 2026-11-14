"""Environment record written beside every result: git commit, package versions, hardware."""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "huggingface_hub",
    "sentence-transformers",
    "peft",
    "safetensors",
    "tokenizers",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def collect_env() -> dict:
    import psutil
    import torch

    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    # Code and configuration only: result and report files written by a run do not make it dirty.
    status = _git("status", "--porcelain", "--", ".", ":(exclude)results", ":(exclude)reports")
    env = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda_build": torch.version.cuda,
        "ram_total_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env["gpus"].append({"index": i, "name": props.name, "vram_total_gib": round(props.total_memory / 1024**3, 2)})
    return env
