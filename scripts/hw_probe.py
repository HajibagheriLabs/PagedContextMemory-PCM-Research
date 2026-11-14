"""Hardware and software probe. Writes a JSON report (default reports/hw_local.json)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GIB = 1024**3


def cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def nvidia_smi() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = "index,name,driver_version,memory.total,memory.used,memory.free,temperature.gpu"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    rows = []
    for line in out.strip().splitlines():
        vals = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(fields.split(","), vals)))
    return rows


def disk(path: Path) -> dict:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "anchor": Path(path).resolve().anchor,
        "total_gib": round(usage.total / GIB, 2),
        "free_gib": round(usage.free / GIB, 2),
    }


def probe() -> dict:
    import psutil
    import torch
    import transformers

    from pcm.hub import hub_cache_dir

    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": platform.python_version(),
        "cpu": {
            "name": cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "ram": {
            "total_gib": round(vm.total / GIB, 2),
            "free_gib": round(vm.available / GIB, 2),
            "swap_total_gib": round(sm.total / GIB, 2),
            "swap_free_gib": round(sm.free / GIB, 2),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(),
        },
        "transformers": transformers.__version__,
        "gpus": [],
        "nvidia_smi": nvidia_smi(),
        "disk": {
            "project": disk(REPO_ROOT),
            "hf_cache": disk(hub_cache_dir()),
        },
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": str(hub_cache_dir()),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "HF_TOKEN_set": bool(os.environ.get("HF_TOKEN")),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_total_gib": round(total / GIB, 2),
                    "vram_free_gib": round(free / GIB, 2),
                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    drivers = {r["driver_version"] for r in info["nvidia_smi"]}
    info["nvidia_driver"] = sorted(drivers)[0] if drivers else None
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "hw_local.json")
    args = ap.parse_args()
    info = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
