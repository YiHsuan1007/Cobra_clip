"""Helpers for applying percentile-based activation clipping."""
from __future__ import annotations

from typing import Iterable, Optional, Union

import torch

ClipValue = Union[float, Iterable[float], torch.Tensor]


def _as_tensor(clip: ClipValue, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(clip, torch.Tensor):
        return clip.to(device=ref.device, dtype=ref.dtype)
    if isinstance(clip, Iterable):
        return torch.as_tensor(list(clip), dtype=ref.dtype, device=ref.device)
    return torch.tensor(float(clip), dtype=ref.dtype, device=ref.device)


def clamp_tensor_(tensor: torch.Tensor, clip: Optional[ClipValue]) -> torch.Tensor:
    """Clamp ``tensor`` in-place using symmetric ``clip`` bounds."""
    if clip is None:
        return tensor
    clip_tensor = _as_tensor(clip, tensor)
    if clip_tensor.numel() == 1:
        max_val = clip_tensor.item()
        return tensor.clamp_(min=-max_val, max=max_val)
    # Broadcast the clip tensor across the trailing dimension when possible.
    clip_tensor = clip_tensor.view(*([1] * (tensor.ndim - 1)), -1)
    return torch.minimum(torch.maximum(tensor, -clip_tensor), clip_tensor)


def clamp_from_state_(tensor: torch.Tensor, state: dict) -> torch.Tensor:
    """Clamp ``tensor`` using a serialized observer state if available."""
    clip = state.get("clip") if isinstance(state, dict) else None
    return clamp_tensor_(tensor, clip)


def load_clip_value(state: dict) -> Optional[ClipValue]:
    """Return the clip value stored inside an observer state dictionary."""
    if not isinstance(state, dict):
        return None
    return state.get("clip")
