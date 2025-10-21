"""Observers for percentile-driven activation statistics."""
from __future__ import annotations

from typing import Dict, Optional, Union

import torch

ClipValue = Union[float, torch.Tensor]


class PercentileObserver:
    """Collect samples and estimate percentile-based clipping thresholds."""

    def __init__(
        self,
        p_max: float,
        mode: str = "tensor",
        max_samples: int = 1_000_000,
        *,
        target: Optional[str] = None,
    ) -> None:
        if not (0.0 < p_max <= 100.0):
            raise ValueError(f"`p_max` must be in (0, 100], received {p_max}.")
        if mode != "tensor":
            raise ValueError(f"Mode `{mode}` is not supported; only `tensor` aggregation is available.")
        if max_samples <= 0:
            raise ValueError("`max_samples` must be positive.")

        self.p_max = float(p_max)
        self.mode = mode
        self.max_samples = int(max_samples)
        self.target = target
        self._samples: Optional[torch.Tensor] = None
        self._numel: int = 0
        self._clip_value: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._samples = None
        self._numel = 0
        self._clip_value = None

    @torch.no_grad()
    def update(self, tensor: torch.Tensor) -> None:
        if tensor is None or self._clip_value is not None:
            return

        data = tensor.detach()
        if data.is_sparse:
            data = data.to_dense()

        values = data.abs().to(torch.float32).reshape(-1).cpu()
        if values.numel() == 0:
            return

        self._numel += int(values.numel())
        if values.numel() > self.max_samples:
            idx = torch.randperm(values.numel())[: self.max_samples]
            values = values.index_select(0, idx)

        if self._samples is None:
            self._samples = values.clone()
        else:
            combined = torch.cat([self._samples, values], dim=0)
            if combined.numel() > self.max_samples:
                idx = torch.randperm(combined.numel())[: self.max_samples]
                combined = combined.index_select(0, idx)
            self._samples = combined

    def _compute_clip(self) -> Optional[torch.Tensor]:
        if self._clip_value is not None:
            return self._clip_value
        if self._samples is None or self._samples.numel() == 0:
            return None
        quantile = torch.quantile(self._samples, self.p_max / 100.0)
        self._clip_value = quantile
        return quantile

    def get_clip_value(self) -> Optional[ClipValue]:
        clip = self._compute_clip()
        if clip is None:
            return None
        return clip.item() if clip.numel() == 1 else clip

    def has_value(self) -> bool:
        return self._compute_clip() is not None

    def state_dict(self) -> Dict[str, Union[float, int, ClipValue, None]]:
        clip = self.get_clip_value()
        if isinstance(clip, torch.Tensor):
            clip_payload: Union[float, list[float]]
            clip_payload = clip.cpu().tolist()
        else:
            clip_payload = clip  # may be ``None``
        payload = {
            "p_max": self.p_max,
            "mode": self.mode,
            "clip": clip_payload,
            "numel": self._numel,
        }
        if self.target is not None:
            payload["target"] = self.target
        return payload

    def load_state_dict(self, state: Dict[str, Union[float, int, ClipValue, None]]) -> None:
        self.reset()
        self.p_max = float(state.get("p_max", self.p_max))
        self.mode = state.get("mode", self.mode)
        if "target" in state and state["target"] is not None:
            self.target = str(state["target"])
        clip = state.get("clip")
        if clip is None:
            return
        if isinstance(clip, torch.Tensor):
            self._clip_value = clip.to(torch.float32)
        elif isinstance(clip, (list, tuple)):
            self._clip_value = torch.as_tensor(clip, dtype=torch.float32)
        else:
            self._clip_value = torch.tensor(float(clip), dtype=torch.float32)

    @property
    def numel(self) -> int:
        return self._numel


class DualModalityObserver:
    """Helper that keeps separate observers for DINO and SigLIP activations."""

    def __init__(self, p_max: float, mode: str = "tensor", max_samples: int = 1_000_000) -> None:
        self.dino = PercentileObserver(p_max=p_max, mode=mode, max_samples=max_samples)
        self.siglip = PercentileObserver(p_max=p_max, mode=mode, max_samples=max_samples)

    def reset(self) -> None:
        self.dino.reset()
        self.siglip.reset()

    def update(self, dino_tensor: torch.Tensor, siglip_tensor: torch.Tensor) -> None:
        self.dino.update(dino_tensor)
        self.siglip.update(siglip_tensor)

    def state_dict(self) -> Dict[str, Dict[str, Union[float, int, ClipValue, None]]]:
        return {"dino": self.dino.state_dict(), "siglip": self.siglip.state_dict()}

    def load_state_dict(self, state: Dict[str, Dict[str, Union[float, int, ClipValue, None]]]) -> None:
        if "dino" in state:
            self.dino.load_state_dict(state["dino"])
        if "siglip" in state:
            self.siglip.load_state_dict(state["siglip"])

    def clip_values(self) -> Dict[str, Optional[ClipValue]]:
        return {"dino": self.dino.get_clip_value(), "siglip": self.siglip.get_clip_value()}

    def has_value(self) -> bool:
        return self.dino.has_value() and self.siglip.has_value()

