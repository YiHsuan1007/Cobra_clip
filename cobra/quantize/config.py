"""Configuration objects for percentile-based quantization utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load percentile configs.") from exc


@dataclass
class QuantConfig:
    """Configuration for percentile-based activation clipping."""

    p_max: float = 99.9
    mode: str = "tensor"
    stats_path: Union[str, Path] = Path("percentile_stats.pt")
    max_samples: int = 1_000_000
    batch_size: int = 8
    num_batches: Optional[int] = None
    prompt: str = "Describe the image in detail."
    num_workers: int = 4
    device: Optional[str] = None
    dtype: Optional[Union[str, torch.dtype]] = None
    targets: Optional[Tuple[str, ...]] = None
    weight_bits: int = 8
    act_bits: int = 8
    act_quant: bool = True
    add_quant: bool = True
    swiglu_quant: bool = True
    swilu_quant: bool = True
    act_quant_params: Dict[str, Any] = field(default_factory=dict)
    x1_quant_params: Dict[str, Any] = field(default_factory=dict)
    x2_quant_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 < self.p_max <= 100.0):
            raise ValueError(f"`p_max` must be in (0, 100], received {self.p_max}.")

        self.mode = self.mode.lower()
        self.stats_path = Path(self.stats_path)
        if self.targets is not None:
            if isinstance(self.targets, (list, tuple)):
                self.targets = tuple(str(t).lower() for t in self.targets)
            else:
                raise TypeError("`targets` must be a sequence of strings when provided.")
        if self.max_samples <= 0:
            raise ValueError("`max_samples` must be positive.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.num_batches is not None and self.num_batches <= 0:
            raise ValueError("`num_batches` must be positive when provided.")
        try:
            self.weight_bits = int(self.weight_bits)
            self.act_bits = int(self.act_bits)
        except (TypeError, ValueError) as exc:
            raise TypeError("`weight_bits` and `act_bits` must be integers.") from exc
        if self.weight_bits <= 0 or self.act_bits <= 0:
            raise ValueError("`weight_bits` and `act_bits` must be positive integers.")
        if not isinstance(self.act_quant_params, dict):
            raise TypeError("`act_quant_params` must be a dictionary.")
        if not isinstance(self.x1_quant_params, dict):
            raise TypeError("`x1_quant_params` must be a dictionary.")
        if not isinstance(self.x2_quant_params, dict):
            raise TypeError("`x2_quant_params` must be a dictionary.")
        if self.dtype is not None:
            try:
                self.dtype = self._normalize_dtype(self.dtype)
            except (AttributeError, ValueError, TypeError) as exc:
                raise TypeError("`dtype` must be a torch.dtype or string dtype name.") from exc

    @property
    def percentile(self) -> float:
        """Return the percentile in the [0, 1] range."""
        return self.p_max / 100.0

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "QuantConfig":
        """Load configuration parameters from a YAML file."""
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as handle:
            payload: Dict[str, Any] = yaml.safe_load(handle) or {}
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QuantConfig":
        """Create a configuration from a dictionary payload."""
        data = dict(payload)
        bits = data.pop("bits", None)
        if isinstance(bits, dict):
            data.setdefault("weight_bits", bits.get("weight", data.get("weight_bits", 8)))
            data.setdefault("act_bits", bits.get("activation", data.get("act_bits", 8)))
        data.setdefault("weight_bits", 8)
        data.setdefault("act_bits", 8)
        # Legacy percentile override fields are ignored now that the feature is removed.
        data.pop("percentile", None)
        data.pop("clipping", None)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation."""
        if isinstance(self.dtype, torch.dtype):
            dtype_value = str(self.dtype).replace("torch.", "")
        else:
            dtype_value = self.dtype
        return {
            "p_max": self.p_max,
            "mode": self.mode,
            "stats_path": str(self.stats_path),
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "prompt": self.prompt,
            "num_workers": self.num_workers,
            "device": self.device,
            "dtype": dtype_value,
            "targets": list(self.targets) if self.targets is not None else None,
            "weight_bits": self.weight_bits,
            "act_bits": self.act_bits,
            "act_quant": self.act_quant,
            "add_quant": self.add_quant,
            "swiglu_quant": self.swiglu_quant,
            "swilu_quant": self.swilu_quant,
            "act_quant_params": self.act_quant_params,
            "x1_quant_params": self.x1_quant_params,
            "x2_quant_params": self.x2_quant_params,
        }

    @staticmethod
    def _normalize_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            token = value.strip()
            if token.startswith("torch."):
                token = token.split(".", 1)[1]
            candidate = getattr(torch, token, None)
            if isinstance(candidate, torch.dtype):
                return candidate
        raise ValueError(f"Unrecognised torch dtype {value!r}.")

