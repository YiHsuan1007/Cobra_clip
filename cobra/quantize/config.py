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
    """Configuration for percentile-based activation clipping.

    Attributes
    ----------
    p_max:
        Upper percentile (0-100) used to determine the clipping threshold.
    mode:
        Aggregation strategy used by the observers. Currently, "tensor" is supported
        which computes a single percentile across the entire tensor. Additional modes can
        be added in the future (e.g. per-channel) by extending the observers.
    stats_path:
        Location where calibration statistics are stored. Relative paths are resolved
        with respect to the current working directory when the configuration is loaded.
    max_samples:
        Maximum number of samples retained by observers while estimating the percentile.
        Larger values improve stability at the cost of memory usage.
    batch_size:
        Batch size used during calibration runs.
    num_batches:
        Optional limit on the number of batches processed during calibration. None
        means that the entire dataloader is consumed.
    prompt:
        Default textual prompt used when running calibration without task specific data.
    num_workers:
        Number of dataloader workers.
    device / dtype:
        Optional device override (e.g. "cuda" or "cpu"). If None the script
        will select cuda when available. ``dtype`` controls the calibration
        tensor dtype and accepts either a torch.dtype instance or a string name.
    weight_bits / act_bits:
        Bit-width used for weight and activation quantisation respectively.
    act_quant:
        Whether activation quantization should be enabled on auxiliary wrappers
        such as QuantSoftmax/QuantAdd.
    add_quant / swiglu_quant / swilu_quant:
        Feature toggles controlling whether the corresponding helper wrappers
        should be instantiated when `replace_other_layers` is executed.
    act_quant_params / x1_quant_params / x2_quant_params:
        Optional dictionaries with quantizer configuration forwarded to the
        respective quantized helper modules.
    """

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

