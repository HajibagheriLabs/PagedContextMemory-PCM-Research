"""Store of page-only KV caches (specification, section 4.5): bf16, on the GPU while they fit, on disk after.

A page cache (KVCache of one page, encoded as a continuation of the shared prefix) is kept as it
is on the GPU until the store holds `gpu_bytes`; later pages go to disk, one file per page: a numpy
array of shape [2, layers, kv_heads, tokens, head_dim] holding the keys and the values bit for bit
(bf16 stored as int16), opened as a memmap when read. Reading a page back gives the same tensors
that were put in.
"""
from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from pcm.store.kv import KVCache

_BITS = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


class PageStore:
    def __init__(self, gpu_bytes: int, disk_dir: Path | None = None, device: str | torch.device = "cuda"):
        self.gpu_bytes = int(gpu_bytes)
        self.device = torch.device(device)
        self._own_dir = disk_dir is None
        self.dir = Path(disk_dir) if disk_dir is not None else None
        self.gpu: dict = {}
        self.disk: dict = {}   # key -> (path, dtype)
        self.bytes_gpu = 0
        self.bytes_disk = 0
        self.write_s = 0.0

    def __contains__(self, key) -> bool:
        return key in self.gpu or key in self.disk

    def __len__(self) -> int:
        return len(self.gpu) + len(self.disk)

    def where(self, key) -> str:
        return "gpu" if key in self.gpu else "disk"

    def _disk_dir(self) -> Path:
        if self.dir is None:
            self.dir = Path(tempfile.mkdtemp(prefix="pcm_pages_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir

    def put(self, key, cache: KVCache) -> str:
        """Store a page cache; returns where it went ("gpu" or "disk")."""
        if key in self:
            raise KeyError(f"page {key!r} is already stored")
        if self.bytes_gpu + cache.nbytes <= self.gpu_bytes:
            self.gpu[key] = cache.to(device=self.device)
            self.bytes_gpu += cache.nbytes
            return "gpu"
        t0 = time.perf_counter()
        dtype = cache.dtype
        if dtype not in _BITS:
            raise TypeError(f"no bit-exact disk format for {dtype}")
        stacked = torch.stack([torch.stack(cache.keys), torch.stack(cache.values)])  # [2, L, 1, H, n, D]
        arr = stacked[:, :, 0].contiguous().view(_BITS[dtype]).cpu().numpy()
        path = self._disk_dir() / f"{len(self.disk):06d}.npy"
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
        mm[...] = arr
        mm.flush()
        del mm
        self.disk[key] = (path, dtype)
        self.bytes_disk += cache.nbytes
        self.write_s += time.perf_counter() - t0
        return "disk"

    def get(self, key) -> KVCache:
        """The page cache on the store's device."""
        if key in self.gpu:
            return self.gpu[key]
        path, dtype = self.disk[key]
        mm = np.load(path, mmap_mode="r")
        arr = np.array(mm, copy=True)  # read the file once; the map is closed before the tensor is used (Windows locks mapped files)
        del mm
        t = torch.from_numpy(arr).to(self.device).view(dtype)  # [2, L, H, n, D]
        n_layers = t.shape[1]
        return KVCache([t[0, i].unsqueeze(0) for i in range(n_layers)], [t[1, i].unsqueeze(0) for i in range(n_layers)])

    def clear(self) -> None:
        self.gpu.clear()
        for path, _ in self.disk.values():
            path.unlink(missing_ok=True)
        self.disk.clear()
        self.bytes_gpu = self.bytes_disk = 0
        if self._own_dir and self.dir is not None:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None
