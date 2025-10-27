"""dtype utilities for calibration stability."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch


_DEFAULT_CALIB_DTYPE = torch.float32


def force_calib_dtype(x: torch.Tensor, target: torch.dtype = _DEFAULT_CALIB_DTYPE) -> torch.Tensor:
    """Return x cast to the calibration dtype if needed."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("force_calib_dtype expects a torch.Tensor")
    if x.dtype == target:
        return x
    return x.to(dtype=target)


@contextmanager
def scoped_no_autocast():
    """Context manager that temporarily disables autocast."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        with torch.cuda.amp.autocast(enabled=False):
            yield
    else:
        try:
            from torch.amp import autocast  # type: ignore
        except (ImportError, AttributeError):
            yield
        else:
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            with autocast(device_type=device_type, enabled=False):
                yield
