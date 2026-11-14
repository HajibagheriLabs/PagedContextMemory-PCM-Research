"""GPU health: preflight checks, recognising a lost device, and telemetry.

A lost GPU is not an experimental failure. Code that runs examples calls
`classify_failure` on an exception: "gpu_lost" means stop now, leave the results
written so far untouched and exit with EXIT_GPU_LOST so the same command can be
rerun after the device is back; "oom" and "error" are ordinary failures that are
recorded for the example and reported.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3
EXIT_GPU_LOST = 75  # "temporary failure": rerun the same command once the GPU is back

_LOST_MARKERS = (
    "cuda error",
    "cudnn error",
    "cublas error",
    "gpu is lost",
    "device lost",
    "unspecified launch failure",
    "no cuda gpus are available",
    "cuda driver",
    "nvml",
    "unable to determine the device handle",
    "an illegal memory access",
)
_TELEMETRY_FIELDS = "timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used,fan.speed"


class GpuLost(RuntimeError):
    """The CUDA device stopped responding or disappeared."""


class PreflightError(RuntimeError):
    """A resource check failed before a run started."""


def nvidia_smi_query(fields: str, timeout: float = 20.0) -> list[list[str]] | None:
    """Rows of `nvidia-smi --query-gpu=fields`, or None if the driver does not answer."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if out.returncode != 0:
        return None
    return [[v.strip() for v in line.split(",")] for line in out.stdout.strip().splitlines() if line.strip()]


def gpu_status() -> dict:
    """Whether the GPU is usable right now, with the reason if not."""
    import torch

    smi = nvidia_smi_query("name,memory.total,memory.free")
    status = {"driver_responds": smi is not None, "cuda_available": False, "ok": False, "reason": None}
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:  # a broken driver can raise here
        status["reason"] = f"torch.cuda.is_available raised: {e}"
        return status
    if not status["driver_responds"]:
        status["reason"] = "nvidia-smi cannot reach the driver"
    elif not status["cuda_available"]:
        status["reason"] = "PyTorch reports no CUDA device"
    else:
        status["ok"] = True
        status["name"] = smi[0][0]
    return status


def classify_failure(exc: BaseException) -> str:
    """"gpu_lost", "oom" or "error" for an exception raised during a run."""
    import torch

    if isinstance(exc, GpuLost):
        return "gpu_lost"
    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
        return "oom"
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _LOST_MARKERS):
        return "gpu_lost"
    # Any other error while the driver has stopped answering is the lost device showing up indirectly.
    if isinstance(exc, RuntimeError) and nvidia_smi_query("name") is None and shutil.which("nvidia-smi"):
        return "gpu_lost"
    return "error"


def preflight(min_free_vram_gib: float = 0.0, min_free_ram_gib: float = 0.0, min_free_disk_gib: float = 0.0, disk_path: Path | None = None) -> dict:
    """Check GPU, free VRAM, RAM and disk before a run. Raises GpuLost or PreflightError."""
    import psutil
    import torch

    status = gpu_status()
    if not status["ok"]:
        raise GpuLost(status["reason"])
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    ram = psutil.virtual_memory().available
    disk = shutil.disk_usage(disk_path or Path.cwd()).free
    report = {
        "gpu": status.get("name"),
        "free_vram_gib": free_vram / GIB,
        "total_vram_gib": total_vram / GIB,
        "free_ram_gib": ram / GIB,
        "free_disk_gib": disk / GIB,
    }
    problems = []
    if report["free_vram_gib"] < min_free_vram_gib:
        problems.append(f"free VRAM {report['free_vram_gib']:.1f} GiB < {min_free_vram_gib} GiB")
    if report["free_ram_gib"] < min_free_ram_gib:
        problems.append(f"free RAM {report['free_ram_gib']:.1f} GiB < {min_free_ram_gib} GiB")
    if report["free_disk_gib"] < min_free_disk_gib:
        problems.append(f"free disk {report['free_disk_gib']:.1f} GiB < {min_free_disk_gib} GiB")
    if problems:
        raise PreflightError("; ".join(problems))
    return report


def interruption_record(exc: BaseException, last_done: str | None = None) -> dict:
    """What to write beside the results when a run stops because the GPU was lost."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": classify_failure(exc),
        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        "driver_responds": nvidia_smi_query("name") is not None,
        "last_completed": last_done,
        "resume": "rerun the same command once nvidia-smi answers; completed work is skipped",
    }


class GpuTelemetry:
    """Samples temperature, power, load, memory and fan to a CSV file while a run is active.

    One row every `interval` seconds, flushed as written, so the last rows before
    a crash survive it. Use as a context manager around a run.
    """

    def __init__(self, path: Path, interval: float = 10.0):
        self.path = Path(path)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.is_file() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["utc"] + _TELEMETRY_FIELDS.split(","))
            while not self._stop.is_set():
                rows = nvidia_smi_query(_TELEMETRY_FIELDS, timeout=10.0)
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                writer.writerow([now] + (rows[0] if rows else ["driver not responding"]))
                f.flush()
                self._stop.wait(self.interval)

    def __enter__(self) -> "GpuTelemetry":
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 15)


def wait_for_gpu(timeout_s: float = 0.0, poll_s: float = 30.0) -> bool:
    """True once the GPU answers. With a timeout, keep polling until it does or time runs out."""
    deadline = time.monotonic() + timeout_s
    while True:
        if gpu_status()["ok"]:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
