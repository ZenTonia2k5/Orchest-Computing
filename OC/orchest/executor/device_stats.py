"""Device-agnostic timing synchronisation and peak-memory readings for stage traces."""

from typing import Optional, Union

import torch

MB = 1024.0 * 1024.0


def _as_device(device: Union[str, torch.device, None]) -> Optional[torch.device]:
    if device is None:
        return None
    return device if isinstance(device, torch.device) else torch.device(device)


def synchronize(device: Union[str, torch.device, None]) -> None:
    """Waits for queued kernels so wall-clock timings cover the actual work."""
    dev = _as_device(device)
    if dev is None:
        return
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    elif dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_memory(device: Union[str, torch.device, None]) -> None:
    dev = _as_device(device)
    if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)


def peak_memory_mb(device: Union[str, torch.device, None]) -> Optional[float]:
    """
    Peak allocated accelerator memory since the last reset (CUDA), or currently allocated
    memory (MPS, which has no peak counter). None on CPU, where it is not tracked.
    """
    dev = _as_device(device)
    if dev is None:
        return None
    if dev.type == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(dev) / MB, 2)
    if dev.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return round(torch.mps.current_allocated_memory() / MB, 2)
    return None


def peak_memory_kind(device: Union[str, torch.device, None]) -> str:
    dev = _as_device(device)
    if dev is not None and dev.type == "cuda":
        return "cuda_max_allocated"
    if dev is not None and dev.type == "mps":
        return "mps_current_allocated"
    return "not_tracked"


def is_oom_error(message: str) -> bool:
    m = (message or "").lower()
    return "out of memory" in m or "outofmemory" in m or "mps backend out of memory" in m
