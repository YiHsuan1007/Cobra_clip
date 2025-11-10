"""Utilities for deriving per-module percentile overrides."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import nn

from ..int_linear import QuantLinear
from ..percentile_aliases import normalize_target_name
from ..quantizer import UniformAffineQuantizer

try:  # pragma: no cover - optional conv wrappers
    from ..int_conv import QuantConv1d, QuantConv2d, QuantConv3d, QuantConvBase
except ImportError:  # pragma: no cover
    QuantConvBase = tuple()  # type: ignore[assignment]
    QuantConv1d = QuantConv2d = QuantConv3d = tuple()  # type: ignore[assignment]

try:  # pragma: no cover - optional policy helper
    from . import percentile_policy  # type: ignore
except Exception:  # pragma: no cover
    percentile_policy = None

PercentileOverrides = Dict[str, Dict[str, Any]]

_DEFAULT_STAGE_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("vision_backbone.dino", ("vision_backbone.dino", "dino_featurizer")),
    ("vision_backbone.siglip", ("vision_backbone.siglip", "siglip_featurizer")),
    ("projector.out", ("projector", "projector.out", "tap_post_mm_out")),
    ("llm_backbone", ("llm_backbone", "language_model", "llm")),
)


@dataclass(frozen=True)
class _StageConfig:
    stage: str
    percentile: float


def load_stats(path: str | Path) -> Mapping[str, Any]:
    """Load percentile statistics from disk."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Percentile stats '{resolved}' not found.")
    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Percentile stats must be a mapping, received {type(payload)!r}.")
    return payload


def _normalise_percentile(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        percentile = float(value)
    except (TypeError, ValueError):
        return None
    if percentile <= 1.0:
        percentile *= 100.0
    return max(0.0, min(100.0, percentile))


def _resolve_stage_from_name(name: str) -> Optional[str]:
    canonical = normalize_target_name(name)
    for stage, aliases in _DEFAULT_STAGE_ALIASES:
        for token in aliases:
            if token in canonical:
                return stage
    for stage, aliases in _DEFAULT_STAGE_ALIASES:
        if canonical.startswith(stage):
            return stage
    return None


def _normalise_stats(stats: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    observers = stats.get("observers") if isinstance(stats, Mapping) else None
    if not isinstance(observers, Mapping):
        return {}
    normalized: Dict[str, Mapping[str, Any]] = {}
    for key, value in observers.items():
        canonical = normalize_target_name(str(key))
        if isinstance(value, Mapping):
            normalized[canonical] = value
    return normalized


def _select_stage_percentile(
    stage: str,
    stats_entry: Optional[Mapping[str, Any]],
    policy: str,
    default_p: float,
) -> float:
    policy = (policy or "auto").lower()
    if policy == "auto":
        if percentile_policy is not None and hasattr(percentile_policy, "select_clip_percent"):
            try:
                selected = percentile_policy.select_clip_percent(stage, stats_entry or {}, default_p)  # type: ignore[attr-defined]
                normalized = _normalise_percentile(selected)
                if normalized is not None:
                    return normalized
            except Exception:
                pass
        if isinstance(stats_entry, Mapping):
            for key in ("percentile", "percent", "p_max"):
                normalized = _normalise_percentile(stats_entry.get(key))
                if normalized is not None:
                    return normalized
    normalized_default = _normalise_percentile(default_p)
    return float(normalized_default if normalized_default is not None else 99.9)


def _iter_quant_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    quant_conv_types = tuple(
        klass for klass in (QuantConv1d, QuantConv2d, QuantConv3d) if isinstance(klass, type)
    )
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            yield name, module
        elif quant_conv_types and isinstance(module, quant_conv_types):
            yield name, module
        elif isinstance(module, QuantConvBase):  # pragma: no cover - safety guard
            yield name, module


def _collect_quantizers(candidate: Any) -> list[UniformAffineQuantizer]:
    quantizers: list[UniformAffineQuantizer] = []
    seen: set[int] = set()

    def _collect(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        if isinstance(obj, UniformAffineQuantizer):
            quantizers.append(obj)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                _collect(value)
            return
        if isinstance(obj, (list, tuple, set)):
            for value in obj:
                _collect(value)
            return
        if isinstance(obj, (nn.ModuleDict, nn.ModuleList, nn.Sequential)):
            for value in obj:
                _collect(value)

    _collect(candidate)
    return quantizers


def build_percentile_overrides(
    model: nn.Module,
    stats: Mapping[str, Any] | None,
    policy: str = "auto",
    default_p: float = 99.9,
    include_kinds: Tuple[str, ...] = ("weight_quantizer", "act_quantizer"),
) -> PercentileOverrides:
    """
    Build per-module percentile overrides derived from stage-level stats.

    Parameters
    ----------
    model:
        Model containing QuantLinear / QuantConv modules.
    stats:
        Stage-level percentile statistics (e.g., outputs/percentile_stats.pt contents).
    policy:
        Override selection policy. ``"auto"`` attempts to call percentile_policy.select_clip_percent.
    default_p:
        Fallback percentile when no stats are available.
    include_kinds:
        Quantizer attributes to include (typically ``weight_quantizer`` / ``act_quantizer``).
    """

    stats_map = _normalise_stats(stats or {})
    overrides: PercentileOverrides = {}
    stage_cache: Dict[str, _StageConfig] = {}

    for module_name, module in _iter_quant_modules(model):
        stage = _resolve_stage_from_name(module_name)
        if stage is None:
            continue
        stage_entry = stage_cache.get(stage)
        if stage_entry is None:
            percentile = _select_stage_percentile(stage, stats_map.get(stage), policy, default_p)
            stage_entry = _StageConfig(stage, percentile)
            stage_cache[stage] = stage_entry
        for attr in include_kinds:
            if not attr:
                continue
            quant_obj = getattr(module, attr, None)
            if not _collect_quantizers(quant_obj):
                continue
            key = f"{module_name}.{attr}"
            overrides[key] = {"percentile": stage_entry.percentile}

    return overrides


def apply_overrides(
    model: nn.Module,
    overrides: str | Mapping[str, Mapping[str, Any]],
) -> int:
    """
    Apply per-module percentile overrides to quantizers.

    Parameters
    ----------
    model:
        Model containing quantized modules.
    overrides:
        Mapping or path to torch-saved overrides.

    Returns
    -------
    int
        Number of quantizer attributes updated.
    """

    if isinstance(overrides, (str, Path)):
        payload = torch.load(Path(overrides).expanduser(), map_location="cpu")
    else:
        payload = overrides

    if not isinstance(payload, Mapping):
        raise TypeError("Overrides must be a mapping or a path to a serialized mapping.")

    named_modules = dict(model.named_modules())
    updated = 0
    linear_hits = 0
    conv_hits = 0
    preview: list[tuple[str, float]] = []
    prefix_blocked: Dict[str, bool] = {entry[0]: False for entry in _DEFAULT_STAGE_ALIASES}

    for key, descriptor in payload.items():
        if not isinstance(key, str) or "." not in key or not isinstance(descriptor, Mapping):
            continue
        module_name, attr = key.rsplit(".", 1)
        module = named_modules.get(module_name)
        if module is None:
            continue
        percentile_value = descriptor.get("percentile", descriptor.get("percent"))
        percentile_float = _normalise_percentile(percentile_value)
        if percentile_float is None:
            continue
        quantizers = _collect_quantizers(getattr(module, attr, None))
        if not quantizers:
            continue
        for quantizer in quantizers:
            if hasattr(quantizer, "percent"):
                quantizer.percent = percentile_float
            if hasattr(quantizer, "percentile"):
                quantizer.percentile = percentile_float
            observer = getattr(quantizer, "observer", None)
            if observer is not None and hasattr(observer, "percent"):
                observer.percent = percentile_float / 100.0 if percentile_float > 1.0 else percentile_float
        updated += 1
        if isinstance(module, QuantLinear):
            linear_hits += 1
        elif isinstance(module, QuantConvBase):
            conv_hits += 1
        if len(preview) < 10:
            preview.append((key, percentile_float))
        stage = _resolve_stage_from_name(module_name)
        if stage in prefix_blocked:
            prefix_blocked[stage] = True

    missing_prefixes = [stage for stage, applied in prefix_blocked.items() if not applied]
    for stage in missing_prefixes:
        logger = logging.getLogger(__name__)
        logger.warning(
            "[PercentileOverrides] No overrides applied for stage prefix: %s", stage
        )
    if preview:
        print("[PercentileOverrides] sample entries:")
        for entry, percentile in preview:
            print(f"  - {entry} -> percentile=p{percentile:.4g}")
    if updated:
        print(f"[QuantPct][apply] overrides_applied={updated} linear={linear_hits} conv={conv_hits}")
    return updated


def save_overrides(path: str | Path, mapping: PercentileOverrides) -> None:
    """Persist overrides to ``path``."""
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mapping), resolved)

