"""Configuration objects for percentile-based quantization utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

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
        Aggregation strategy used by the observers. Currently, ``"tensor"`` is supported
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
        Optional limit on the number of batches processed during calibration. ``None``
        means that the entire dataloader is consumed.
    prompt:
        Default textual prompt used when running calibration without task specific data.
    num_workers:
        Number of dataloader workers.
    device:
        Optional device override (e.g. ``"cuda"`` or ``"cpu"``). If ``None`` the script
        will select ``cuda`` when available.
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
    targets: Optional[Tuple[str, ...]] = None

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

    @property
    def percentile(self) -> float:
        """Return the percentile in the [0, 1] range."""
        return self.p_max / 100.0

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "QuantConfig":
        """Load configuration parameters from a YAML file."""
        cfg_path = Path(path)
        with cfg_path.open("r", encoding="utf-8") as f:
            payload: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary representation."""
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
            "targets": list(self.targets) if self.targets is not None else None,
        }
