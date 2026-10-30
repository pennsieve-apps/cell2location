"""
How many CPUs this container may use, and where PyTorch runs.

os.cpu_count() reports the host (or VM) CPUs, which inside a CPU-limited container (Docker
--cpus, an ECS/Fargate task) can be several times the allocation. THREADS=0 ("one per vCPU")
is therefore resolved from the cgroup CPU quota (v2 cpu.max, v1 cpu.cfs_quota_us /
cpu.cfs_period_us), else the CPU affinity, else os.cpu_count().

USE_GPU=auto takes cuda when torch sees a GPU (the gpu runtime) and cpu otherwise, so the same
image serves computeTypes [standard, gpu]; USE_GPU=true on a host without a GPU falls back to
cpu with a warning rather than failing.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_cpus(root: Path = CGROUP) -> float | None:
    """CPUs allowed by the cgroup CPU quota, or None when there is no quota."""
    v2 = _read(root / "cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max":
            try:
                return int(quota) / int(period or 100000)
            except (ValueError, ZeroDivisionError):
                return None
        return None
    for d in (root / "cpu", root / "cpu,cpuacct", root):
        q, p = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
        if q and p:
            try:
                q_, p_ = int(q), int(p)
            except ValueError:
                return None
            return q_ / p_ if q_ > 0 and p_ > 0 else None
    return None


def cgroup_memory_bytes(root: Path = CGROUP) -> int | None:
    """The cgroup memory limit in bytes, or None when there is none."""
    for f in (root / "memory.max", root / "memory" / "memory.limit_in_bytes", root / "memory.limit_in_bytes"):
        v = _read(f)
        if v is None:
            continue
        if v == "max":
            return None
        try:
            n = int(v)
        except ValueError:
            return None
        return n if 0 < n < (1 << 60) else None
    return None


def cgroup_memory_peak_mb(root: Path = CGROUP) -> float | None:
    """cgroup v2 memory.peak (the container's high-water mark, page cache included), in MB."""
    v = _read(root / "memory.peak")
    try:
        return round(int(v) / 2**20, 1) if v else None
    except ValueError:
        return None


def available_cpus(root: Path = CGROUP) -> tuple[int, str]:
    """Whole CPUs this process may use (at least 1) and where the number comes from."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota = cgroup_cpus(root)
    if quota is not None:
        n = max(1, math.floor(quota + 1e-9))
        if affinity:
            n = min(n, affinity)
        return n, f"cgroup CPU quota {quota:g}"
    if affinity:
        return affinity, "CPU affinity"
    return max(1, os.cpu_count() or 1), "os.cpu_count()"


def resolve_threads(requested: int) -> tuple[int, str]:
    """THREADS as given when > 0, else one per available CPU."""
    if requested > 0:
        return requested, f"THREADS={requested}"
    return available_cpus()


def set_thread_env(n: int) -> None:
    """Pin the BLAS / OpenMP / numba thread pools; must run before numpy is imported."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


DEVICES = ("auto", "true", "false")


def normalise_use_gpu(value: str) -> str:
    v = (value or "auto").strip().lower()
    if v in ("1", "yes", "cuda", "gpu", "on"):
        return "true"
    if v in ("0", "no", "cpu", "off"):
        return "false"
    return v


def choose_device(use_gpu: str) -> dict:
    """{requested, accelerator ('cpu' or 'gpu'), reason[, gpu, cuda, cudnn]} for USE_GPU."""
    requested = normalise_use_gpu(use_gpu)
    if requested == "false":
        return {"requested": requested, "accelerator": "cpu", "reason": "USE_GPU=false: using cpu"}
    try:
        import torch
    except ImportError:
        return {"requested": requested, "accelerator": "cpu", "reason": f"USE_GPU={requested}: torch is not importable, using cpu"}
    try:
        if torch.cuda.is_available():
            return {"requested": requested, "accelerator": "gpu",
                    "reason": f"USE_GPU={requested}: using cuda ({torch.cuda.get_device_name(0)})",
                    "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version()}
        why = "cuda not available"
    except Exception as exc:  # a broken CUDA runtime must not take the step down
        why = f"cuda check failed ({exc})"
    return {"requested": requested, "accelerator": "cpu", "reason": f"USE_GPU={requested}: {why}, using cpu"}
