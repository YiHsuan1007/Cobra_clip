"""Runtime control helpers for percentile clipping and quantization toggles."""
from __future__ import annotations

import csv
import functools
import json
import logging
import numbers
import re
import types
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import inspect
import torch
from torch import nn
from torch.utils.data import DataLoader

from cobra.integration.hooks import (
    DEFAULT_PERCENTILE_TARGET_MAP,
    DEFAULT_PERCENTILE_TARGETS,
    attach_percentile_hooks,
    remove_handles,
)
from cobra.quantize.calibrate import calibrate_model, load_stats
from cobra.quantize.config import QuantConfig
from cobra.quantize.int_linear import QuantLinear
try:
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d, QuantConvBase
except ImportError:  # pragma: no cover - fallback when QuantConvBase is unavailable
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d
    QuantConvBase = (QuantConv1d, QuantConv2d, QuantConv3d)  # type: ignore[assignment]
from cobra.quantize.int_matmul import QuantMatMul, QuantMatmulWrapper
from cobra.quantize.int_others import QuantAdd, QuantSwiglu, QuantSwilu, QuantSoftmax
from cobra.quantize.observers import PercentileObserver, build_observer
from cobra.quantize.rotations import apply_wht_then_klt, compute_klt_from_stats, fold_rotation_into_linear
from cobra.quantize.percentile_aliases import (
    candidate_observer_keys,
    expand_target_for_hooks,
    expand_targets_for_hooks,
    has_hook_targets,
    normalize_target_name,
    normalize_targets,
)
from cobra.quantize.utils import (
    convert_to_int as _convert_to_int_utils,
    enable_observation as _enable_observation_utils,
    finalize_all_quantizers as _finalize_all_quantizers,
    iter_named_modules,
    set_observing,
    set_quant_state,
    set_static_quant,
    _flatten_quantizer_objects,
)
from cobra.quantize.quantizer import UniformAffineQuantizer
from cobra.quantize.utils.dtype import force_calib_dtype, scoped_no_autocast

__all__ = [
    "calibrate",
    "calibrate_percentiles",
    "calibrate_quantization",
    "enable",
    "disable",
    "replace_linear_layers",
    "replace_conv_layers",
    "replace_matmul_layers",
    "replace_other_layers",
    "enable_quant",
    "disable_all_quant",
    "convert_to_int",
    "activate_observers",
    "finalize_quant_params",
    "wrap_model_for_percentile",
"rewrite_percentile_module_path",
]

_HANDLE_ATTR = "_quant_pct_handles"
_OBSERVER_ATTR = "_quant_pct_observers"

ActivationCallback = Callable[[str, str, object], None]

DEFAULT_TARGETS: Sequence[str] = tuple(DEFAULT_PERCENTILE_TARGETS)

_DEFAULT_NORMALIZED_ORDER: Tuple[str, ...] = (
    "vision_backbone.dino",
    "vision_backbone.siglip",
    "llm_backbone",
    "projector",
)

_CANONICAL: Set[str] = {
    "vision_backbone",
    *_DEFAULT_NORMALIZED_ORDER,
}

_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "vision": ("vision_backbone",),
    "vb": ("vision_backbone",),
    "dino": ("vision_backbone.dino",),
    "siglip": ("vision_backbone.siglip",),
    "llm": ("llm_backbone",),
    "mm": ("projector",),
    "projector.out": ("projector",),
}


def _normalize_targets(raw: Iterable[str]) -> List[str]:
    out: Set[str] = set()
    unknown: Set[str] = set()

    def _resolve(token: str) -> None:
        queue: List[str] = [token]
        seen: Set[str] = set()
        while queue:
            key = (queue.pop() or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            if key == "vision_backbone":
                out.update(("vision_backbone.dino", "vision_backbone.siglip"))
                continue
            if key in _ALIAS_MAP:
                queue.extend(_ALIAS_MAP[key])
                continue
            if key.startswith("vision_backbone."):
                out.add(key)
                continue
            if key in _CANONICAL:
                out.add(key)
                continue
            unknown.add(key)

    for entry in raw:
        text = (entry or "").strip()
        if not text:
            continue
        parts = [p.strip() for p in text.split(",")] if "," in text else [text]
        for part in parts:
            if not part:
                continue
            _resolve(part)

    if not out:
        out.update(_DEFAULT_NORMALIZED_ORDER)

    ordered = [name for name in _DEFAULT_NORMALIZED_ORDER if name in out]
    extras = sorted(out.difference(_DEFAULT_NORMALIZED_ORDER))
    ordered.extend(extras)
    if unknown:
        logging.warning(
            "[QuantPct] Ignoring unknown percentile target(s): %s",
            ", ".join(sorted(unknown)),
        )
    return ordered


LEGACY_TARGET_MAP: Dict[str, str] = {
    # Early keys
    "vision.dino": "vision_backbone.dino_featurizer",
    "vision.siglip": "vision_backbone.siglip_featurizer",
    "mm.out": "projector.out",
    # Newly exported keys that still need expansion
    "vision_backbone.dino": "vision_backbone.dino_featurizer",
    "vision_backbone.siglip": "vision_backbone.siglip_featurizer",
    "projector.out": "projector.out",
}

_FINAL_TARGET_PREFIX: Dict[str, str] = {
    "vision_backbone.dino": "vision_backbone.dino_featurizer",
    "vision_backbone.siglip": "vision_backbone.siglip_featurizer",
    "projector.out": "projector.out",
}


def _build_module_prefix_remap() -> Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...]:
    remap: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    for hook_name, module_path in DEFAULT_PERCENTILE_TARGET_MAP.items():
        canonical = normalize_target_name(hook_name)
        final_prefix = _FINAL_TARGET_PREFIX.get(canonical)
        if final_prefix is None:
            continue
        source_segments = tuple(part for part in module_path.split(".") if part)
        target_segments = tuple(part for part in final_prefix.split(".") if part)
        if not source_segments or not target_segments:
            continue
        remap.append((source_segments, target_segments))
    return tuple(remap)


_MODULE_PREFIX_REMAP = _build_module_prefix_remap()


def rewrite_percentile_module_path(path: str) -> str:
    """
    Normalize module paths so percentile stats export/import share consistent prefixes.

    When the model nests target modules under an extra namespace (e.g. ``model.vision_backbone``),
    this helper rewrites the matching sub-paths to the canonical prefixes used during export.
    """

    if not path:
        return path
    segments = [part for part in path.split(".") if part]
    if not segments:
        return path

    for source_segments, target_segments in _MODULE_PREFIX_REMAP:
        length = len(source_segments)
        if length == 0 or length > len(segments):
            continue
        for offset in range(len(segments) - length + 1):
            if segments[offset : offset + length] == list(source_segments):
                rewritten = segments[:offset] + list(target_segments) + segments[offset + length :]
                return ".".join(rewritten)
    return path
logger = logging.getLogger(__name__)

_Batch = Any
_Args = Tuple[Any, ...]
_Kwargs = Dict[str, Any]
_ForwardExtractor = Callable[[Any], Tuple[_Args, _Kwargs]]

try:
    _QUANT_LINEAR_PARAMS = inspect.signature(QuantLinear.__init__).parameters
    _QUANT_LINEAR_SUPPORTS_BITS = {"weight_bits", "act_bits"}.issubset(_QUANT_LINEAR_PARAMS.keys())
except (TypeError, ValueError):
    _QUANT_LINEAR_SUPPORTS_BITS = True


def _normalize_target_prefixes(targets: Optional[Sequence[Any]]) -> Optional[Tuple[str, ...]]:
    if targets is None:
        return None
    normalized: List[str] = []
    for item in targets:
        candidate: Optional[str] = None
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            for key in ("module", "name", "target", "path"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    break
        elif hasattr(item, "name"):
            value = getattr(item, "name", None)
            if isinstance(value, str):
                candidate = value.strip()
        if candidate is None:
            candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized) or None


def _normalize_target_list(targets: Sequence[str]) -> List[str]:
    return list(normalize_targets(targets))


def _format_quantizer_key(module_name: str, role: str, idx: int, total: int) -> str:
    """
    Return the canonical percentile key used for a quantizer attribute.

    ``module_name`` must be the dotted path reported by ``named_modules()``.
    ``role`` is normalised to ``weight`` or ``act``.
    ``idx`` enumerates quantizers when multiple exist under the same attribute.
    """

    if total <= 0:
        raise ValueError("`total` must be positive when formatting quantizer keys.")
    if idx < 0 or idx >= total:
        raise IndexError(f"Quantizer index {idx} is out of range for total={total}.")

    normalized_role = "weight" if str(role).lower().startswith("weight") else "act"
    return f"{module_name}.{normalized_role}_quantizer.{idx}"


def _iter_percentile_quantizers(module: nn.Module, attr: str) -> List[UniformAffineQuantizer]:
    return [
        quant
        for quant in _flatten_quantizer_objects(getattr(module, attr, None))
        if isinstance(quant, UniformAffineQuantizer)
        and str(getattr(quant, "mode", "")).lower() == "percentile"
    ]


def _replace_module(root: nn.Module, path: str, new_module: nn.Module) -> None:
    if not path:
        raise ValueError("Cannot replace the root module.")
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _iter_named_modules_flex(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield `(qualified_name, module)` whether named_modules returns pairs or triplets."""
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    for entry in named_modules():
        if not isinstance(entry, (list, tuple)):
            continue
        if len(entry) == 2:
            name, mod = entry
            yield name, mod
        elif len(entry) == 3:
            parent, name, mod = entry
            qualified = f"{parent}.{name}" if parent else name
            yield qualified, mod
        else:
            continue


def _safe_named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Yield (name, module) even if named_modules returns extra metadata."""
    yield from _iter_named_modules_flex(model)


def iter_percentile_quantizers(
    model: nn.Module,
) -> Iterable[Tuple[str, str, UniformAffineQuantizer]]:
    """Yield `(path, role, quantizer)` for every percentile-mode quantizer."""
    for module_name, module in _safe_named_modules(model):
        if not module_name:
            continue
        weight_quant = getattr(module, "weight_quantizer", None)
        if isinstance(weight_quant, UniformAffineQuantizer) and str(
            getattr(weight_quant, "mode", "")
        ).lower() == "percentile":
            yield module_name, "weight", weight_quant
        act_quant = getattr(module, "act_quantizer", None)
        if isinstance(act_quant, UniformAffineQuantizer) and str(
            getattr(act_quant, "mode", "")
        ).lower() == "percentile":
            yield module_name, "act", act_quant


def count_percentile_quantizers(model: nn.Module) -> Tuple[int, int]:
    weight = 0
    act = 0
    samples: List[Tuple[str, str, Any]] = []
    for path, role, quant in iter_percentile_quantizers(model):
        if len(samples) < 5:
            samples.append((path, role, getattr(quant, "mode", None)))
        if role == "weight":
            weight += 1
        else:
            act += 1
    print(f"[Debug][Quantizer][samples]={samples}")
    return weight, act


def export_percentile_quantizers(
    model: nn.Module,
    stats_path: Optional[str | Path] = None,
) -> Tuple[int, OrderedDict[str, Dict[str, Any]]]:
    entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    exported = 0
    for module_name, role, quant in iter_percentile_quantizers(model):
        exporter = getattr(quant, "export_percentile_stats", None)
        if not callable(exporter):
            continue
        payload = exporter()
        if not payload:
            continue
        rewritten = rewrite_percentile_module_path(module_name)
        key = f"{rewritten}.{role}_quantizer"
        entries[key] = dict(payload)
        exported += 1

    if stats_path:
        resolved = Path(stats_path).expanduser()
        if resolved.exists():
            existing = torch.load(resolved, map_location="cpu")
            if not isinstance(existing, dict):
                raise TypeError(f"Percentile stats at {resolved} must be a mapping.")
            base: Dict[str, Any] = dict(existing)
        else:
            base = {}
        base.update(entries)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base, resolved)

    return exported, entries


def _load_and_apply_percentile_stats(model: nn.Module, path: str, logger: logging.Logger) -> Tuple[int, int]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        logger.warning("[PercentileStats] File not found: %s", resolved)
        return 0, 0
    try:
        payload = torch.load(resolved, map_location="cpu")
    except Exception as exc:
        logger.warning("[PercentileStats] Failed to load %s: %s", resolved, exc)
        return 0, 0
    if not isinstance(payload, Mapping):
        logger.warning("[PercentileStats] Payload at %s is not a mapping.", resolved)
        return 0, 0

    payload = dict(payload)
    for key in list(payload.keys()):
        if not isinstance(key, str) or "::" not in key:
            continue
        head, _, tail = key.partition("::")
        if head in {"target", "observer"} and tail and tail not in payload:
            payload[tail] = payload[key]

    applied = 0
    missing = 0
    for module_name, module in _safe_named_modules(model):
        if not module_name:
            continue
        canonical_module = rewrite_percentile_module_path(module_name)
        for role, attr in (("weight", "weight_quantizer"), ("act", "act_quantizer")):
            quantizers = _iter_percentile_quantizers(module, attr)
            total = len(quantizers)
            if total == 0:
                continue
            for idx, quantizer in enumerate(quantizers):
                key = _format_quantizer_key(canonical_module, role, idx, total)
                stats_entry = payload.get(key)
                if stats_entry is None and canonical_module != module_name:
                    legacy_key = _format_quantizer_key(module_name, role, idx, total)
                    stats_entry = payload.get(legacy_key)
                    if stats_entry is not None:
                        key = legacy_key
                if isinstance(stats_entry, Mapping):
                    try:
                        quantizer.apply_percentile_stats(stats_entry)
                        applied += 1
                        continue
                    except Exception as exc:
                        logger.debug("[PercentileStats] Failed to apply %s: %s", key, exc)
                missing += 1

    message = f"[PercentileStats] Applied {applied} quantizer entries from {resolved} (missing={missing})"
    logger.info(message)
    print(message)
    return applied, missing


def _should_wrap_name(name: str, prefixes: Optional[Sequence[str]], substring: bool = False) -> bool:
    if not prefixes:
        return True
    if substring:
        return any(prefix in name for prefix in prefixes)
    return any(name.startswith(prefix) for prefix in prefixes)


def _log_replacement_summary(
    kind: str,
    names: Sequence[str],
    max_expected: Optional[int],
    raise_on_excess: bool,
    *,
    budget_from_targets: bool = False,
) -> None:
    count = len(names)
    preview = ", ".join(names[:10])
    suffix = f": {preview}" if preview else ""
    logging.info("[QuantPct][%s] Replaced %d module(s)%s", kind, count, suffix)
    if max_expected is None:
        return
    if budget_from_targets and count > max_expected:
        max_expected = count
    if count > max_expected:
        message = (
            f"[QuantPct][{kind}] Replaced {count}; calibrated on {max_expected} nodes; proceed to replay to populate stats."
        )
        if raise_on_excess:
            raise RuntimeError(message)
        logging.info(message)


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _normalize_stat_keys(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Harmonise percentile statistics payload keys to match observer expectations.

    - Promote ``target::<name>`` keys to ``observer::<name>`` entries and bare ``<name>`` keys.
    - Augment entries with canonical and legacy aliases so downstream lookups succeed.
    - Preserve original values without mutation.
    """

    if not isinstance(stats, dict):
        return stats

    def _alias_variants(name: str) -> List[str]:
        variants: Set[str] = set()
        raw = name.strip()
        if raw:
            variants.add(raw)
        canonical = normalize_target_name(raw)
        if canonical:
            variants.add(canonical)
        for legacy_key, mapped in LEGACY_TARGET_MAP.items():
            if raw == legacy_key or raw == mapped or canonical == legacy_key or canonical == mapped:
                variants.add(legacy_key)
                variants.add(mapped)
        return [entry for entry in variants if entry]

    target_key_values: Dict[str, Any] = {}
    for key in list(stats.keys()):
        if not isinstance(key, str) or not key.startswith("target::"):
            continue
        suffix = key.split("target::", 1)[1]
        value = stats[key]
        target_key_values[suffix] = value
        for alias in _alias_variants(suffix):
            observer_key = f"observer::{alias}"
            if observer_key not in stats:
                stats[observer_key] = value
            if alias not in stats:
                stats[alias] = value

    normalized_observers: Dict[str, Any] = {}
    observers_obj = stats.get("observers")
    if isinstance(observers_obj, Mapping):
        source_items = observers_obj.items()
    else:
        source_items = ()
    for key, value in source_items:
        if isinstance(key, str):
            if key.startswith("target::"):
                suffix = key.split("target::", 1)[1]
            elif key.startswith("observer::"):
                suffix = key.split("observer::", 1)[1]
            else:
                suffix = key
        else:
            suffix = str(key)
        for alias in _alias_variants(suffix):
            normalized_observers.setdefault(alias, value)
            prefixed = f"observer::{alias}"
            normalized_observers.setdefault(prefixed, value)

    for suffix, value in target_key_values.items():
        for alias in _alias_variants(suffix):
            normalized_observers.setdefault(alias, value)
            prefixed = f"observer::{alias}"
            normalized_observers.setdefault(prefixed, value)

    if normalized_observers:
        stats["observers"] = normalized_observers

    return stats


# Percentile clipping helpers -------------------------------------------------------------------

def calibrate_percentiles(
    model,
    dataloader: DataLoader,
    cfg: QuantConfig,
    targets: Optional[Iterable[str]] = None,
) -> dict:
    """Run calibration and persist observer statistics."""
    stats = calibrate_model(model, dataloader, cfg, targets=targets)
    _finalize_pending_percentiles(model, logger)
    return stats


def activate_observers(model: nn.Module) -> None:
    """Enable observation on quantization wrappers."""
    _enable_observation_utils(model)


def finalize_quant_params(model: nn.Module) -> None:
    """Finalize quantizer parameters prior to real-int execution."""
    _finalize_all_quantizers(model)
    try:
        summaries, _ = _finalize_model_quantizers(model, {}, {})
    except Exception:
        logger.debug("[QuantPct] Failed to emit calibration summaries during finalize_quant_params.", exc_info=True)
        return
    for line in summaries:
        print(line)


def _build_observer(state: dict, cfg: QuantConfig, target: str) -> PercentileObserver:
    observer = PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples, target=target)
    observer.load_state_dict(state)
    return observer

def normalize_stats_format(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("Percentile stats must be a mapping.")
    if "targets" in raw and "observers" in raw:
        return raw

    pattern = re.compile(r"^target::(.+)$")
    observer_entries: Dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        match = pattern.match(key)
        if match is None:
            continue
        canonical = normalize_target_name(match.group(1))
        observer_entries[canonical] = value

    if not observer_entries:
        raise ValueError("Unsupported stats format")

    normalized_targets = sorted(observer_entries.keys())
    return {"targets": normalized_targets, "observers": observer_entries}

def _collect_observers(
    stats: dict,
    cfg: QuantConfig,
) -> Tuple[Dict[str, PercentileObserver], List[Tuple[str, List[str], List[str]]]]:
    stats = normalize_stats_format(stats)
    observers = stats.get("observers", {})
    if not isinstance(observers, Mapping):
        observers = {}
    registered_targets = normalize_targets(stats.get("targets") or DEFAULT_TARGETS)
    expanded_targets: List[str] = []
    for target in registered_targets:
        descendants = expand_target_for_hooks(target)
        if descendants:
            expanded_targets.extend(descendants)
        else:
            expanded_targets.append(target)
    lookup_targets = _normalize_target_list(expanded_targets)
    result: Dict[str, PercentileObserver] = {}
    missing: List[Tuple[str, List[str], List[str]]] = []
    available_keys = [str(key) for key in observers.keys()]
    for name in lookup_targets:
        state = None
        candidates = list(candidate_observer_keys(name))
        for candidate in candidates:
            if candidate in observers:
                state = observers[candidate]
                break
        if state is None:
            canonical_name = normalize_target_name(name)
            missing.append((canonical_name, candidates, available_keys))
            continue
        canonical_name = normalize_target_name(name)
        result[canonical_name] = _build_observer(state, cfg, target=canonical_name)
    return result, missing

def wrap_model_for_percentile(
    model: nn.Module,
    weight_bits: int,
    act_bits: int,
    percent: float,
) -> nn.Module:
    """Ensure supported modules are wrapped with percentile-ready quantizers."""
    if getattr(model, "_quant_pct_fake_wrapped", False):
        return model

    if percent > 1.0:
        percent = percent / 100.0

    cfg_stub = types.SimpleNamespace(
        weight_bits=weight_bits,
        act_bits=act_bits,
        targets=None,
        observe="percentile",
        conv_observer="percentile",
        matmul_observer="percentile",
        conv_weight_quant_params=None,
        conv_act_quant_params=None,
        matmul_x1_quant_params=None,
        matmul_x2_quant_params=None,
        disable_input_quant=False,
        conv_weight_bits=weight_bits,
        conv_act_bits=act_bits,
        matmul_act_bits=act_bits,
        strict_target_budget=False,
    )

    replace_linear_layers(model, cfg_stub, weight_bits=weight_bits, act_bits=act_bits)
    replace_conv_layers(model, cfg_stub, weight_bits=weight_bits, act_bits=act_bits)
    replace_matmul_layers(model, cfg_stub, act_bits=act_bits)

    for module in model.modules():
        for attr in ("weight_quantizer", "act_quantizer"):
            quantizers = _flatten_quantizer_objects(getattr(module, attr, None))
        for quantizer in quantizers:
            if not isinstance(quantizer, UniformAffineQuantizer):
                continue
            quantizer.mode = "percentile"
            quantizer.percent = percent
            quantizer.observered = False
            quantizer.cached_xmin = None
            quantizer.cached_xmax = None
            if hasattr(quantizer, "observer") and quantizer.observer is not None:
                quantizer.observer.owner = quantizer
            if hasattr(quantizer, "scale"):
                quantizer.scale = None
            if hasattr(quantizer, "round_zero_point"):
                quantizer.round_zero_point = None
            quantizer._pending_percentile = True
    setattr(model, "_quant_pct_fake_wrapped", True)
    return model


def _finalize_pending_percentiles(
    model: nn.Module,
    logger: logging.Logger,
) -> None:
    for module_name, module in _safe_named_modules(model):
        for role, attr in (("weight", "weight_quantizer"), ("act", "act_quantizer")):
            quantizers = _iter_percentile_quantizers(module, attr)
            total = len(quantizers)
            if total == 0:
                continue
            for idx, quantizer in enumerate(quantizers):
                pending = getattr(quantizer, "_pending_percentile", False)
                if not pending:
                    continue
                stats_payload = quantizer.export_percentile_stats()
                if not isinstance(stats_payload, Mapping):
                    canonical_module = rewrite_percentile_module_path(module_name)
                    key = _format_quantizer_key(canonical_module, role, idx, total)
                    logger.warning("[PercentileStats][warn] missing observer for %s", key)
                    continue
                quantizer.apply_percentile_stats(stats_payload)


def _configure_quantizer_percentile(
    quantizer: Optional[UniformAffineQuantizer],
    percent: float,
) -> None:
    if quantizer is None:
        return
    quantizer.mode = "percentile"
    quantizer.percent = float(percent)
    quantizer.cached_xmin = None
    quantizer.cached_xmax = None
    quantizer.scale = None
    quantizer.round_zero_point = None
    quantizer._pending_percentile = True  # type: ignore[attr-defined]
    if hasattr(quantizer, "observer"):
        observer = getattr(quantizer, "observer")
        if observer is not None:
            observer.owner = quantizer  # type: ignore[attr-defined]
    if hasattr(quantizer, "observered"):
        quantizer.observered = False


def _wrap_for_collect(model: nn.Module, cfg: QuantConfig) -> None:
    from torch import nn as torch_nn

    percent = float(getattr(cfg, "p_max", 0.999))
    weight_bits = int(getattr(cfg, "weight_bits", 8))
    act_bits = int(getattr(cfg, "act_bits", 8))

    def _prepare_quantizer_container(module: nn.Module) -> None:
        _configure_quantizer_percentile(getattr(module, "weight_quantizer", None), percent)
        _configure_quantizer_percentile(getattr(module, "act_quantizer", None), percent)
        setattr(module, "_quant_pct_collect_wrapped", True)

    for name, module in list(_iter_named_modules_flex(model)):
        if not name:
            continue
        if getattr(module, "_quant_pct_collect_wrapped", False):
            continue
        if isinstance(module, QuantLinear):
            _prepare_quantizer_container(module)
            continue
        if isinstance(module, QuantConv2d):
            _prepare_quantizer_container(module)
            continue
        if isinstance(module, torch_nn.Linear):
            quant_layer = QuantLinear(
                module,
                observe="percentile",
                weight_bits=weight_bits,
                act_bits=act_bits,
            )
            _prepare_quantizer_container(quant_layer)
            _replace_module(model, name, quant_layer)
            continue
        if isinstance(module, torch_nn.Conv2d):
            weight_params = {
                "dynamic_method": "per_tensor",
                "n_bits": weight_bits,
                "shape": module.weight.shape,
                "is_weight": True,
                "observe": "percentile",
            }
            act_params = {
                "dynamic_method": "per_tensor",
                "n_bits": act_bits,
                "has_batch_dim": True,
                "observe": "percentile",
            }
            quant_layer = QuantConv2d(
                module,
                weight_quant_params=weight_params,
                act_quant_params=act_params,
                observe="percentile",
            )
            _prepare_quantizer_container(quant_layer)
            _replace_module(model, name, quant_layer)


def enable(
    model,
    cfg: QuantConfig,
    *,
    mode: str = "apply",
    dumper: Optional[ActivationCallback] = None,
    targets: Optional[Iterable[str]] = None,
    strict_missing_stats: bool = False,
    wrap_fake_quant: bool = False,
    fake_quant_bits: Optional[Tuple[int, int]] = None,
    export_per_quant: bool = False,
    amp: Optional[bool] = None,
) -> None:
    """Enable percentile clipping by registering forward hooks on ``model``."""

    if targets is not None:
        raw_targets: Iterable[str]
        if isinstance(targets, str):
            raw_targets = [targets]
        else:
            raw_targets = list(targets)
    else:
        cfg_targets = getattr(cfg, "targets", None)
        if cfg_targets is None:
            raw_targets = []
        elif isinstance(cfg_targets, str):
            raw_targets = [cfg_targets]
        else:
            raw_targets = list(cfg_targets)

    normalized_targets = _normalize_targets(raw_targets)
    setattr(cfg, "targets_normalized", normalized_targets)
    print(f"[QuantPct] Normalized targets: {normalized_targets}")
    if export_per_quant:
        setattr(cfg, "_quant_pct_export_per_quant", True)
    if amp is not None:
        setattr(cfg, "_quant_pct_amp_request", bool(amp))

    requested_targets: Sequence[str] = tuple(normalized_targets)

    canonical_requested = tuple(_normalize_target_list(requested_targets))
    hook_targets_sequence = expand_targets_for_hooks(canonical_requested)
    hook_targets = tuple(hook_targets_sequence)

    skipped_targets: List[str] = []
    for name in canonical_requested:
        canonical = normalize_target_name(name)
        expanded = tuple(normalize_targets(expand_target_for_hooks(name)))
        if expanded or has_hook_targets(canonical):
            continue
        skipped_targets.append(canonical)
    if skipped_targets:
        logger.warning(
            "[QuantPct] Skipping targets without hook mapping: %s",
            ", ".join(skipped_targets),
        )

    normalized_mode = mode.lower()
    if normalized_mode not in {"apply", "collect", "off"}:
        raise ValueError("`mode` must be one of {\"apply\", \"collect\", \"off\"}.")

    if normalized_mode == "off":
        disable(model)
        return

    disable(model)

    if normalized_mode == "collect":
        _wrap_for_collect(model, cfg)
    elif wrap_fake_quant:
        fq_bits = fake_quant_bits or (cfg.weight_bits, cfg.act_bits)
        percent_value = cfg.p_max
        wrap_model_for_percentile(model, fq_bits[0], fq_bits[1], percent_value)

    observer_map: Dict[str, PercentileObserver]
    if normalized_mode == "collect":
        observer_map = {}
        handles: List[Any] = []
    else:  # apply
        stats = load_stats(cfg.stats_path)
        stats = normalize_stats_format(stats)
        targets_preview = stats.get("targets", [])
        logger.info(
            "[QuantPct] Loaded stats targets=%d preview=%s",
            len(targets_preview),
            ", ".join(list(targets_preview)[:10]),
        )
        cfg_from_stats = stats.get("config", {})
        cfg.p_max = cfg_from_stats.get("p_max", cfg.p_max)
        cfg.mode = cfg_from_stats.get("mode", cfg.mode)
        cfg.max_samples = cfg_from_stats.get("max_samples", cfg.max_samples)

        observer_map_all, missing_details = _collect_observers(stats, cfg)
        percentile_applied = percentile_missing = 0
        stats_path_value = getattr(cfg, "stats_path", None)
        if stats_path_value:
            percentile_applied, percentile_missing = _load_and_apply_percentile_stats(
                model, stats_path_value, logger
            )

        expected_canonical = [normalize_target_name(name) for name in hook_targets]
        available_canonical = set(observer_map_all.keys())
        missing_required = [name for name in expected_canonical if name not in available_canonical]
        hits = len(expected_canonical) - len(missing_required)
        missing_preview = ", ".join(missing_required[:10]) if missing_required else "<none>"
        print(
            "[QuantPct][apply] observer coverage: "
            f"required={len(expected_canonical)} available={hits} missing={len(missing_required)} "
            f"missing_preview={missing_preview}"
        )
        if stats_path_value:
            print(
                f"[QuantPct][apply] percentile stats: applied={percentile_applied} missing={percentile_missing}"
            )

        if missing_required:
            stats_source = getattr(cfg, "stats_path", None)
            if strict_missing_stats:
                details_map = {entry[0]: entry[1:] for entry in missing_details}
                diagnostic_segments = []
                for missing_name in missing_required:
                    candidates, available_keys = details_map.get(missing_name, ([], []))
                    candidate_preview = ", ".join(candidates[:8]) if candidates else "<none>"
                    available_preview = ", ".join(available_keys[:12]) if available_keys else "<none>"
                    diagnostic_segments.append(
                        f"{missing_name} (candidates={candidate_preview}; available={available_preview})"
                    )
                detail_msg = "; ".join(diagnostic_segments)
                raise KeyError(
                    f"Percentile stats missing required observers for {missing_required} "
                    f"(source={stats_source}). Diagnostics: {detail_msg}"
                )
            else:
                warnings.warn(
                    "[QuantPct] Proceeding despite missing percentile observers: "
                    f"{', '.join(missing_required[:10])}",
                    RuntimeWarning,
                )

        available_hook_targets = [
            name for name in hook_targets if normalize_target_name(name) in available_canonical
        ]
        if not available_hook_targets:
            raise KeyError(
                "No percentile observers matched the requested hook targets after normalization "
                f"(source={getattr(cfg, 'stats_path', None)})."
            )

        observer_map = {
            name: observer_map_all[normalize_target_name(name)]
            for name in available_hook_targets
        }
        handles = attach_percentile_hooks(
            model,
            observers=observer_map,
            apply_clipping=True,
            targets=available_hook_targets,
            dumper=dumper,
        )
    setattr(model, _HANDLE_ATTR, handles)
    setattr(model, _OBSERVER_ATTR, observer_map)


def disable(model) -> None:
    """Disable percentile clipping hooks if they are registered."""
    handles = getattr(model, _HANDLE_ATTR, None)
    if handles is not None:
        remove_handles(handles)
        delattr(model, _HANDLE_ATTR)
    if hasattr(model, _OBSERVER_ATTR):
        delattr(model, _OBSERVER_ATTR)

"""
# Quantization switchboard helpers --------------------------------------------------------------
"""
def _make_quant_linear(
    base_module: nn.Linear,
    *,
    weight_bits: int,
    act_bits: int,
    weight_quant_params: Optional[Dict[str, Any]] = None,
    act_quant_params: Optional[Dict[str, Any]] = None,
    disable_input_quant: bool = False,
    observe: Optional[str] = None,
) -> QuantLinear:
    """Instantiate ``QuantLinear`` with backward-compatible keyword handling."""

    resolved_weight_bits = 8 if weight_bits is None else int(weight_bits)
    resolved_act_bits = 8 if act_bits is None else int(act_bits)

    weight_params = dict(weight_quant_params or {})
    act_params = dict(act_quant_params or {})

    common_kwargs: Dict[str, Any] = {
        "weight_quant_params": weight_params or None,
        "act_quant_params": act_params or None,
        "disable_input_quant": disable_input_quant,
    }
    if observe is not None:
        common_kwargs["observe"] = observe

    if _QUANT_LINEAR_SUPPORTS_BITS:
        quant_layer = QuantLinear(
            base_module,
            weight_bits=resolved_weight_bits,
            act_bits=resolved_act_bits,
            **common_kwargs,
        )
    else:
        if not weight_params:
            weight_params = {"dynamic_method": "per_tensor"}
        weight_params.setdefault("n_bits", resolved_weight_bits)
        weight_params.setdefault("shape", base_module.weight.shape)
        weight_params.setdefault("is_weight", True)

        if disable_input_quant:
            act_params = {}
        elif not act_params:
            act_params = {"dynamic_method": "per_tensor"}
        if act_params:
            act_params.setdefault("n_bits", resolved_act_bits)
            act_params.setdefault("has_batch_dim", True)

        fallback_kwargs: Dict[str, Any] = {
            "weight_quant_params": weight_params,
            "act_quant_params": act_params,
            "disable_input_quant": disable_input_quant,
        }
        if observe is not None:
            fallback_kwargs["observe"] = observe

        quant_layer = QuantLinear(
            base_module,
            **fallback_kwargs,
        )

    if not hasattr(quant_layer, "weight_bits"):
        quant_layer.weight_bits = resolved_weight_bits  # type: ignore[attr-defined]
    if not hasattr(quant_layer, "act_bits"):
        quant_layer.act_bits = resolved_act_bits  # type: ignore[attr-defined]
    if not hasattr(quant_layer, "_origin_linear"):
        object.__setattr__(quant_layer, "_origin_linear", base_module)  # type: ignore[attr-defined]
        quant_layer._modules.pop("_origin_linear", None)

    setattr(quant_layer, "_quant_pct_wrapped", True)

    return quant_layer


def _make_quant_conv(
    base_module: nn.Module,
    *,
    weight_bits: int,
    act_bits: int,
    weight_quant_params: Optional[Dict[str, Any]] = None,
    act_quant_params: Optional[Dict[str, Any]] = None,
    disable_input_quant: bool = False,
    observe: Optional[str] = None,
) -> nn.Module:
    """Instantiate a quantized convolution module matching ``base_module``."""
    if isinstance(base_module, (QuantConv1d, QuantConv2d, QuantConv3d)):
        return base_module
    if not isinstance(base_module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise TypeError(f"_make_quant_conv expects an nn.Conv module, received {type(base_module)!r}.")

    resolved_weight_bits = 8 if weight_bits is None else int(weight_bits)
    resolved_act_bits = 8 if act_bits is None else int(act_bits)

    weight_params = dict(weight_quant_params or {"dynamic_method": "per_tensor"})
    weight_params.setdefault("n_bits", resolved_weight_bits)
    weight_params.setdefault("shape", base_module.weight.shape)
    weight_params.setdefault("is_weight", True)
    if observe is not None:
        weight_params.setdefault("observe", observe)

    act_params = dict(act_quant_params or {"dynamic_method": "per_tensor"})
    act_params.setdefault("n_bits", resolved_act_bits)
    act_params.setdefault("has_batch_dim", True)
    if observe is not None:
        act_params.setdefault("observe", observe)

    observe_mode = weight_params.get("observe", act_params.get("observe", observe or "minmax"))

    if isinstance(base_module, nn.Conv1d):
        quant_layer = QuantConv1d(
            base_module,
            weight_quant_params=weight_params,
            act_quant_params=act_params,
            observe=observe_mode,
            disable_input_quant=disable_input_quant,
        )
    elif isinstance(base_module, nn.Conv2d):
        quant_layer = QuantConv2d(
            base_module,
            weight_quant_params=weight_params,
            act_quant_params=act_params,
            disable_input_quant=disable_input_quant,
            observe=observe_mode,
        )
    else:
        quant_layer = QuantConv3d(
            base_module,
            weight_quant_params=weight_params,
            act_quant_params=act_params,
            disable_input_quant=disable_input_quant,
            observe=observe_mode,
        )

    quant_layer.weight_bits = resolved_weight_bits  # type: ignore[attr-defined]
    quant_layer.act_bits = resolved_act_bits  # type: ignore[attr-defined]
    # Keep a handle to the original conv without registering it as a child to avoid wrapper<->origin cycles.
    object.__setattr__(quant_layer, "_origin_conv", base_module)  # type: ignore[attr-defined]
    quant_layer._modules.pop("_origin_conv", None)
    quant_layer._quant_pct_wrapped = True  # type: ignore[attr-defined]
    return quant_layer


def _make_quant_matmul(
    base_module: nn.Module,
    *,
    act_bits: int,
    x1_quant_params: Optional[Dict[str, Any]] = None,
    x2_quant_params: Optional[Dict[str, Any]] = None,
    observe: Optional[str] = None,
    disable_act_quant: bool = False,
    use_act_quant: bool = False,
) -> QuantMatMul:
    """Instantiate ``QuantMatMul`` mirroring the behaviour of ``base_module``."""
    x1_params = dict(x1_quant_params or {"dynamic_method": "per_tensor"})
    x1_params.setdefault("n_bits", act_bits)
    x1_params.setdefault("has_batch_dim", True)
    if observe is not None:
        x1_params.setdefault("observe", observe)

    x2_params = dict(x2_quant_params or {"dynamic_method": "per_tensor"})
    x2_params.setdefault("n_bits", act_bits)
    x2_params.setdefault("has_batch_dim", True)
    if observe is not None:
        x2_params.setdefault("observe", observe)

    observe_mode = observe or x1_params.get("observe", x2_params.get("observe", "minmax"))
    matmul_func = getattr(base_module, "torch_fn", torch.matmul)
    quant_layer = QuantMatmulWrapper(
        base_module,
        x1_quant_params=x1_params,
        x2_quant_params=x2_params,
        disable_act_quant=disable_act_quant,
        observe=observe_mode,
        matmul_func=matmul_func,
    )
    quant_layer.act_bits = act_bits  # type: ignore[attr-defined]
    quant_layer.use_act_quant = use_act_quant and not disable_act_quant
    setattr(quant_layer, "_quant_pct_wrapped", True)
    return quant_layer


def _is_quant_wrapped(module: nn.Module) -> bool:
    wrapper_cls = globals().get("QuantMatmulWrapper")
    if wrapper_cls is not None and isinstance(module, wrapper_cls):
        return True
    return bool(getattr(module, "_quant_wrapped", False))


def _mark_quant_wrapped(module: nn.Module) -> None:
    setattr(module, "_quant_wrapped", True)


def _matches_torch_matmul(candidate: Any) -> bool:
    if candidate is None:
        return False
    if candidate is torch.matmul:
        return True
    if isinstance(candidate, functools.partial):
        return _matches_torch_matmul(candidate.func)
    wrapped = getattr(candidate, "__wrapped__", None)
    if wrapped is not None and wrapped is not candidate:
        return _matches_torch_matmul(wrapped)
    name = getattr(candidate, "__name__", "") or ""
    qual = getattr(candidate, "__qualname__", "") or ""
    module_name = getattr(candidate, "__module__", "") or ""
    identifier = " ".join((name, qual, module_name)).lower()
    return "matmul" in identifier


def _is_matmul_module(module: nn.Module) -> bool:
    wrapper_cls = globals().get("QuantMatmulWrapper")
    if _is_quant_wrapped(module) or (wrapper_cls is not None and isinstance(module, wrapper_cls)):
        return False
    if isinstance(module, QuantMatMul):
        return False
    cls_name = module.__class__.__name__.lower()
    if "matmul" in cls_name:
        return True
    if _matches_torch_matmul(getattr(module, "torch_fn", None)):
        return True
    for attr_name in (
        "functional",
        "function",
        "fn",
        "op",
        "callable",
        "call_fn",
        "base_fn",
        "inner_fn",
        "matmul_fn",
        "matmul",
    ):
        if _matches_torch_matmul(getattr(module, attr_name, None)):
            return True
    return False


def _classify_matmul_candidate(module: nn.Module) -> str:
    if isinstance(module, (nn.Linear, QuantLinear)):
        return "linear"
    if _is_matmul_module(module):
        return "matmul"
    return "other"


def replace_linear_layers(
    module: nn.Module,
    cfg,
    *,
    weight_bits: Optional[int] = None,
    act_bits: Optional[int] = None,
    targets: Optional[List[str]] = None,
    max_expected: Optional[int] = None,
    raise_on_excess: bool = False,
) -> List[str]:
    """
    Recursively replace nn.Linear modules with QuantLinear instances.

    The helper expects ``cfg`` to expose ``weight_bits`` / ``act_bits`` attributes, but callers
    can customise the values explicitly via the keyword arguments.
    """

    explicit_targets = list(targets) if targets is not None else None

    resolved_weight_bits = int(
        weight_bits if weight_bits is not None else getattr(cfg, "weight_bits", getattr(cfg, "wbits", 8))
    )
    resolved_act_bits = int(
        act_bits if act_bits is not None else getattr(cfg, "act_bits", getattr(cfg, "abits", 8))
    )

    cfg_targets = _get_cfg_value(cfg, "targets")
    prefixes = _normalize_target_prefixes(explicit_targets if explicit_targets is not None else cfg_targets)
    budget_from_targets = False
    if prefixes is not None and max_expected is None:
        max_expected = len(prefixes)
        budget_from_targets = True
    strict_budget = raise_on_excess or bool(_get_cfg_value(cfg, "strict_target_budget", False))

    match_prefixes: Optional[List[str]] = list(prefixes) if prefixes is not None else None

    sample_paths: List[str] = []
    unmatched_samples: List[str] = []

    def _record(path: str) -> None:
        if len(sample_paths) >= 5:
            return
        if path in sample_paths:
            return
        sample_paths.append(path)

    def _match(path: str) -> bool:
        if not match_prefixes:
            return True
        return any(path.startswith(prefix) for prefix in match_prefixes)

    replacements: List[str] = []

    def _walk(current: nn.Module, prefix: str) -> None:
        for child_name, child in list(current.named_children()):
            fq_name = f"{prefix}.{child_name}" if prefix else child_name
            _record(fq_name)
            if getattr(child, "_quant_pct_wrapped", False):
                continue
            if isinstance(child, QuantLinear):
                continue
            if not _match(fq_name):
                continue
            if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
                if _should_wrap_name(fq_name, prefixes):
                    quant_layer = _make_quant_linear(
                        child,
                        weight_bits=resolved_weight_bits,
                        act_bits=resolved_act_bits,
                    )
                    setattr(current, child_name, quant_layer)
                    replacements.append(fq_name)
                    continue
            _walk(child, fq_name)

    _walk(module, "")
    sample_preview = ", ".join(sample_paths) if sample_paths else "<none>"
    print(f"[QuantPct][Linear] sample module paths: {sample_preview}")
    _log_replacement_summary(
        "Linear",
        replacements,
        max_expected,
        strict_budget,
        budget_from_targets=budget_from_targets,
    )
    return replacements

# 10/27新增：替換其他層的量化包裝版本


def _collect_conv_targets(
    root: nn.Module,
    match_prefixes: Optional[Sequence[str]],
    sample_paths: Optional[List[str]] = None,
) -> List[Tuple[nn.Module, str, nn.Conv2d, str]]:
    """Depth-first search for unwrapped Conv2d modules."""
    targets: List[Tuple[nn.Module, str, nn.Conv2d, str]] = []
    visited: Set[int] = set()

    def _record(path: str) -> None:
        if sample_paths is None:
            return
        if len(sample_paths) >= 5:
            return
        if path in sample_paths:
            return
        sample_paths.append(path)

    def _match(path: str) -> bool:
        if not match_prefixes:
            return True
        return any(path.startswith(prefix) for prefix in match_prefixes)

    def _dfs(module: nn.Module, path: str) -> None:
        module_id = id(module)
        if module_id in visited:  # visited prevents revisiting the same module through different parents.
            return
        visited.add(module_id)

        for name, child in module.named_children():
            if getattr(child, "_quant_pct_wrapped", False):
                fq_name = f"{path}.{name}" if path else name
                _record(fq_name)
                continue
            if isinstance(child, QuantConvBase):
                continue
            fq_name = f"{path}.{name}" if path else name
            _record(fq_name)
            if not _match(fq_name):
                continue
            if isinstance(child, nn.Conv2d):
                targets.append((module, name, child, fq_name))
                continue
            _dfs(child, fq_name)

    _dfs(root, "")
    return targets


def replace_conv_layers(
    module: nn.Module,
    cfg,
    *,
    weight_bits: Optional[int] = None,
    act_bits: Optional[int] = None,
    disable_input_quant: Optional[bool] = None,
    observe: Optional[str] = None,
    targets: Optional[List[str]] = None,
    max_expected: Optional[int] = None,
    raise_on_excess: bool = False,
) -> List[str]:
    """Replace convolution modules with quantized variants."""

    explicit_targets = list(targets) if targets is not None else None

    resolved_weight_bits = int(
        weight_bits if weight_bits is not None else getattr(cfg, "weight_bits", getattr(cfg, "wbits", 8))
    )
    resolved_act_bits = int(
        act_bits if act_bits is not None else getattr(cfg, "act_bits", getattr(cfg, "abits", 8))
    )
    disable_inputs = disable_input_quant if disable_input_quant is not None else bool(
        getattr(cfg, "disable_input_quant", False)
    )
    observe_mode = observe or getattr(cfg, "conv_observer", getattr(cfg, "observe", None))

    weight_qp = getattr(cfg, "conv_weight_quant_params", None)
    if weight_qp is None:
        weight_qp = getattr(cfg, "weight_quant_params", None)
    act_qp = getattr(cfg, "conv_act_quant_params", None)
    if act_qp is None:
        act_qp = getattr(cfg, "act_quant_params", getattr(cfg, "a_quant_params", None))

    # Collect once and swap in-place so we never recurse into freshly wrapped layers.
    cfg_targets = _get_cfg_value(cfg, "targets")
    prefixes = _normalize_target_prefixes(explicit_targets if explicit_targets is not None else cfg_targets)
    budget_from_targets = False
    if prefixes is not None and max_expected is None:
        max_expected = len(prefixes)
        budget_from_targets = True
    strict_budget = raise_on_excess or bool(_get_cfg_value(cfg, "strict_target_budget", False))

    match_prefixes: Optional[List[str]] = list(prefixes) if prefixes is not None else None
    sample_paths: List[str] = []

    replacements: List[str] = []
    conv_targets = _collect_conv_targets(module, match_prefixes, sample_paths)
    for parent, name, conv, fq_name in conv_targets:
        if not _should_wrap_name(fq_name, prefixes):
            continue
        quant_layer = _make_quant_conv(
            conv,
            weight_bits=resolved_weight_bits,
            act_bits=resolved_act_bits,
            weight_quant_params=weight_qp,
            act_quant_params=act_qp,
            disable_input_quant=disable_inputs,
            observe=observe_mode,
        )
        setattr(parent, name, quant_layer)
        replacements.append(fq_name)

    sample_preview = ", ".join(sample_paths) if sample_paths else "<none>"
    print(f"[QuantPct][Conv] sample module paths: {sample_preview}")

    _ensure_triplet_named_modules(module)
    _log_replacement_summary(
        "Conv",
        replacements,
        max_expected,
        strict_budget,
        budget_from_targets=budget_from_targets,
    )
    return replacements


def replace_matmul_layers(
    module: nn.Module,
    cfg: Any,
    *,
    act_bits: Optional[int] = None,
    disable_act_quant: Optional[bool] = None,
    observe: Optional[str] = None,
    substring_match: bool = False,
    visited: Optional[Set[int]] = None,
    targets: Optional[List[str]] = None,
    max_expected: Optional[int] = None,
    raise_on_excess: bool = False,
) -> int:
    """
    Recursively replace MatMul helpers with ``QuantMatMul`` and return the replacement count.

    Set ``substring_match=True`` to treat target filters as substring matches instead of strict prefixes.
    """

    explicit_targets = list(targets) if targets is not None else None

    resolved_act_bits = int(act_bits if act_bits is not None else getattr(cfg, "act_bits", getattr(cfg, "abits", 8)))
    disable_outputs = disable_act_quant if disable_act_quant is not None else bool(
        getattr(cfg, "matmul_disable_act_quant", False)
    )
    observe_mode = observe or getattr(cfg, "matmul_observer", getattr(cfg, "observe", None))

    x1_qp = getattr(cfg, "matmul_x1_quant_params", None)
    if x1_qp is None:
        x1_qp = getattr(cfg, "x1_quant_params", None)
    x2_qp = getattr(cfg, "matmul_x2_quant_params", None)
    if x2_qp is None:
        x2_qp = getattr(cfg, "x2_quant_params", None)

    use_act_quant = bool(
        getattr(cfg, "matmul_act_quant", getattr(cfg, "act_quant", getattr(cfg, "enable_act_quant", False)))
    )

    traversal_visited: Set[int]
    if visited is None:
        traversal_visited = set()
    else:
        traversal_visited = set(visited)

    cfg_targets = _get_cfg_value(cfg, "targets")
    prefixes = _normalize_target_prefixes(explicit_targets if explicit_targets is not None else cfg_targets)
    budget_from_targets = False
    if prefixes is not None and max_expected is None:
        max_expected = len(prefixes)
        budget_from_targets = True
    strict_budget = raise_on_excess or bool(_get_cfg_value(cfg, "strict_target_budget", False))

    use_substring = bool(substring_match)
    match_prefixes: Optional[List[str]] = list(prefixes) if prefixes is not None else None
    sample_paths: List[str] = []
    unmatched_samples: list[str] = []
    replaced = 0

    def _append_unmatched(path: str) -> None:
        if len(unmatched_samples) >= 20:
            return
        if path in unmatched_samples:
            return
        unmatched_samples.append(path)

    def _record(path: str) -> None:
        if len(sample_paths) >= 5:
            return
        if path in sample_paths:
            return
        sample_paths.append(path)

    def _match(path: str, substring: bool = False) -> bool:
        if not match_prefixes:
            return True
        if substring:
            return any(prefix in path for prefix in match_prefixes)
        return any(path.startswith(prefix) for prefix in match_prefixes)

    targets_list: List[Tuple[nn.Module, str, nn.Module, str]] = []

    def _dfs(current: nn.Module, path: str) -> None:
        for name, child in list(current.named_children()):
            child_id = id(child)
            if child_id in traversal_visited:
                continue
            traversal_visited.add(child_id)

            if getattr(child, "_quant_pct_wrapped", False):
                continue
            if _is_quant_wrapped(child):
                continue
            if isinstance(child, QuantMatMul):
                continue

            fq_name = f"{path}.{name}" if path else name
            _record(fq_name)

            if not _match(fq_name, use_substring):
                _append_unmatched(fq_name)
                _dfs(child, fq_name)
                continue

            candidate_kind = _classify_matmul_candidate(child)
            if candidate_kind == "matmul":
                if _should_wrap_name(fq_name, prefixes, substring=use_substring):
                    targets_list.append((current, name, child, fq_name))
                    continue
                _append_unmatched(fq_name)
                _dfs(child, fq_name)
                continue
            if candidate_kind == "linear":
                # Attention blocks often alias matmul as Linear; let the linear pass handle them.
                _append_unmatched(f"{fq_name} (linear path handled)")
                _dfs(child, fq_name)
                continue
            _dfs(child, fq_name)

    _dfs(module, "")

    for parent, name, target, fq_name in targets_list:
        quant_layer = _make_quant_matmul(
            target,
            act_bits=resolved_act_bits,
            x1_quant_params=x1_qp,
            x2_quant_params=x2_qp,
            observe=observe_mode,
            disable_act_quant=disable_outputs,
            use_act_quant=use_act_quant,
        )
        _mark_quant_wrapped(quant_layer)
        setattr(parent, name, quant_layer)
        replaced += 1
    sample_preview = ", ".join(sample_paths) if sample_paths else "<none>"
    print(f"[QuantPct][MatMul] sample module paths: {sample_preview}")
    if replaced == 0 and prefixes and any(p in {"vision_backbone", "llm_backbone", "projector"} for p in prefixes):
        logger.info("[QuantPct][MatMul] Checked prefixes cover tap_post_mm_out")
    if max_expected is not None:
        effective_budget = max_expected
        if budget_from_targets and replaced > effective_budget:
            effective_budget = replaced
        if replaced > effective_budget:
            message = (
                f"[QuantPct][MatMul] Replaced {replaced}; calibrated on {effective_budget} nodes; "
                "proceed to replay to populate stats."
            )
            if strict_budget:
                raise RuntimeError(message)
            logger.info(message)
    logger.info("[QuantPct][MatMul] Replaced %d module(s)", replaced)
    if len(unmatched_samples) > 0:
        logger.info("[QuantPct][MatMul] Unmatched samples: %s", ", ".join(unmatched_samples))
    if replaced == 0:
        logger.info("[MatMulQuant] Skipped (attention uses Linear path)")
    if visited is not None:
        visited.clear()
        visited.update(traversal_visited)
    return replaced

def replace_other_layers(module: nn.Module, cfg: Any, visited: Optional[Set[int]] = None) -> None:
    """
    遞迴替換「其他」層為量化包裝版本：
      - nn.Softmax            -> QuantSoftmax
      - *Add / ResidualAdd*   -> QuantAdd        （以類名關鍵字偵測）
      - *SwiGLU / Swiglu*     -> QuantSwiglu     （以類名關鍵字偵測）
      - *Swilu / SiLU-GLU*    -> QuantSwilu      （以類名關鍵字偵測）

    cfg 需可提供：
      - act_quant_params: dict
      - x1_quant_params: dict
      - x2_quant_params: dict
      - act_quant: bool  （是否啟用 activation 量化）
      - add_quant: bool  （是否替換 Add 類模組）
      - swiglu_quant: bool
      - swilu_quant: bool
    若不存在則採用安全預設。
    """

    if visited is None:
        visited = set()
    key = id(module)
    if key in visited:
        return
    visited.add(key)

    # 讀取參數與開關（提供後備鍵以兼容不同命名）
    act_qp: Dict[str, Any] = dict(getattr(cfg, "act_quant_params", getattr(cfg, "a_quant_params", {})))
    x1_qp:  Dict[str, Any] = dict(getattr(cfg, "x1_quant_params", {}))
    x2_qp:  Dict[str, Any] = dict(getattr(cfg, "x2_quant_params", {}))

    use_act_quant: bool = bool(getattr(cfg, "act_quant", getattr(cfg, "enable_act_quant", False)))
    enable_add:   bool = bool(getattr(cfg, "add_quant", True))
    enable_swgl:  bool = bool(getattr(cfg, "swiglu_quant", True))
    enable_swlu:  bool = bool(getattr(cfg, "swilu_quant", True))

    def _is_add_like(m: nn.Module) -> bool:
        # 以類名關鍵字偵測各式「加法/殘差」模組
        name = m.__class__.__name__.lower()
        # 避免把我們自己的 QuantAdd 再次替換
        if isinstance(m, QuantAdd):
            return False
        return any(k in name for k in ("add", "residualadd", "skipadd"))

    def _is_swiglu_like(m: nn.Module, name: str) -> bool:
        if isinstance(m, QuantSwiglu):
            return False
        cls_name = m.__class__.__qualname__.lower()
        nm = name.lower()
        return ("swiglu" in nm) or ("swiglu" in cls_name) or ("swi" in cls_name and "glu" in cls_name)

    def _is_swilu_like(m: nn.Module) -> bool:
        name = m.__class__.__name__.lower()
        if isinstance(m, QuantSwilu):
            return False
        # 常見寫法：Swilu、SiLU-GLU、SwiLU 等
        return ("swilu" in name) or ("silu" in name and "glu" in name)

    def _rebind_module_aliases(parent: nn.Module, original: nn.Module, updated: nn.Module, primary_name: str) -> None:
        modules_dict = getattr(parent, "_modules", None)
        if not isinstance(modules_dict, dict):
            return
        for alias, ref in list(modules_dict.items()):
            if alias == primary_name:
                continue
            if ref is original:
                modules_dict[alias] = updated

    children = list(module.named_children())
    for child_name, listed_child in children:
        modules_dict = getattr(module, "_modules", None)
        if isinstance(modules_dict, dict):
            current_child = modules_dict.get(child_name)
        else:
            current_child = getattr(module, child_name, None)
        if current_child is None:
            continue

        child = current_child
        original_child = child
        replaced = False

        # 1) Softmax -> QuantSoftmax
        if isinstance(child, nn.Softmax):
            qsoft = QuantSoftmax(act_quant_params=act_qp, dim=child.dim, base_module=child)
            qsoft.use_act_quant = use_act_quant
            setattr(module, child_name, qsoft)
            child = qsoft
            replaced = True

        # 2) Add-like -> QuantAdd（僅在有顯式 Add 類模組時可替換；對原生 x+y 無法攔截）
        elif enable_add and _is_add_like(child):
            qadd = QuantAdd(x1_quant_params=x1_qp, x2_quant_params=x2_qp, base_module=child)
            qadd.use_act_quant = use_act_quant
            setattr(module, child_name, qadd)
            child = qadd
            replaced = True

        # 3) SwiGLU-like -> QuantSwiglu
        elif enable_swgl and _is_swiglu_like(child, child_name):
            qswg = QuantSwiglu(x1_quant_params=x1_qp, x2_quant_params=x2_qp, base_module=child)
            qswg.use_act_quant = use_act_quant
            setattr(module, child_name, qswg)
            child = qswg
            replaced = True

        # 4) Swilu-like -> QuantSwilu
        elif enable_swlu and _is_swilu_like(child):
            qswl = QuantSwilu(x1_quant_params=x1_qp, x2_quant_params=x2_qp, base_module=child)
            qswl.use_act_quant = use_act_quant
            setattr(module, child_name, qswl)
            child = qswl
            replaced = True

        if replaced and child is not original_child:
            _rebind_module_aliases(module, original_child, child, child_name)

        # 繼續向下遞迴
        if getattr(child, "_quant_pct_wrapped", False):
            continue
        if child_name.startswith(("_origin", "_original", "_wrapped", "base_module")):
            continue
        if getattr(child, "_is_cobra_internal", False):
            continue
        if child is module:
            continue
        if getattr(child, "_parent_ref", None) is module:
            continue
        replace_other_layers(child, cfg, visited)


def enable_quant(model: nn.Module, **cfg: Any) -> nn.Module:
    """Replace supported layers with their quantized counterparts."""

    weight_quant_params = dict(cfg.get("weight_quant_params", {"dynamic_method": "per_tensor"}))
    act_quant_params = dict(cfg.get("act_quant_params", {"dynamic_method": "per_tensor"}))
    disable_input_quant = cfg.get("disable_input_quant", False)

    observer_cfg = _build_observer_config(cfg)

    if observer_cfg["weight"].get("per_channel_axes") is not None:
        weight_quant_params.setdefault("per_channel_axes", observer_cfg["weight"]["per_channel_axes"])
    act_quant_params.setdefault("per_channel_axes", observer_cfg["activation"].get("per_channel_axes", []))

    observe_token = observer_cfg["activation"]["name"]

    def _resolve_bits(value_map: Dict[str, Any], primary: str, *aliases: str, default: int = 8) -> int:
        for key in (primary, *aliases):
            candidate = value_map.get(key)
            if candidate is not None:
                return int(candidate)
        bits_block = value_map.get("bits")
        if isinstance(bits_block, dict):
            for key in ("weight", "weights") if primary == "weight_bits" else ("activation", "act", "activations"):
                candidate = bits_block.get(key)
                if candidate is not None:
                    return int(candidate)
        return default

    weight_bits = _resolve_bits(cfg, "weight_bits", "wbits")
    act_bits = _resolve_bits(cfg, "act_bits", "abits", "activation_bits")

    for parent, name, child in _walk_named_children(model):
        if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
            quant_layer = _make_quant_linear(
                child,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params,
                disable_input_quant=disable_input_quant,
                observe=observe_token,
                weight_bits=weight_bits,
                act_bits=act_bits,
            )
            setattr(parent, name, quant_layer)
            child = quant_layer

        if isinstance(child, QuantLinear):
            _configure_linear_observers(child, observer_cfg)

    set_quant_state(
        model,
        weight_quant=cfg.get("weight_quant", False),
        act_quant=cfg.get("act_quant", False),
    )
    set_observing(model, observing=cfg.get("observing", False))
    if "static_quant" in cfg:
        set_static_quant(model, static_quant=cfg["static_quant"])

    return model


def calibrate_quantization(
    model: nn.Module,
    data_iter: Iterable[_Batch],
    **cfg: Any,
) -> None:
    """Run calibration passes to collect observer statistics.

    The function keeps quantization disabled while observers collect
    statistics in full precision and restores the original dtype after the
    sweep finishes.
    """

    calibration_dtype = cfg.get("calibration_dtype", torch.float32)
    extractor: Optional[_ForwardExtractor] = cfg.get("forward_extractor")
    max_steps: Optional[int] = cfg.get("max_steps")

    observer_cfg = _build_observer_config(cfg)
    _configure_model_observers(model, observer_cfg)

    log_root = Path(str(cfg.get("quant_log_dir", "runs/quant_logs")))
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = log_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    summary_records: list[Dict[str, Any]] = []

    rotation_mode = cfg.get("rotation_mode", "klt")
    if isinstance(rotation_mode, str):
        rotation_mode = rotation_mode.lower()
    enable_rotation = cfg.get("enable_rotation", True) and rotation_mode in {"klt", "wht+klt"}
    rotation_hooks = []
    rotation_stats: Dict[str, _RotationAccumulator] = {}
    if enable_rotation:
        for name, module in _identify_rotation_targets(model, cfg):
            accumulator = _RotationAccumulator(name, module)
            rotation_stats[name] = accumulator
            rotation_hooks.append(
                module.register_forward_hook(_make_rotation_hook(accumulator))
            )

    activation_hooks = []
    activation_accumulators: Dict[str, _ActivationAccumulator] = {}
    for name, module in iter_named_modules(model):
        if isinstance(module, QuantLinear):
            accumulator = _ActivationAccumulator(name)
            activation_accumulators[name] = accumulator
            activation_hooks.append(module.register_forward_pre_hook(_make_activation_hook(accumulator)))

    dtype_snapshot = _snapshot_tensor_state(model)
    training_mode = model.training

    model.to(calibration_dtype)
    model.eval()

    set_quant_state(
        model,
        weight_quant=cfg.get("calibrate_weight_quant", False),
        act_quant=cfg.get("calibrate_act_quant", True),
    )

    set_observing(model, True)
    try:
        step_iter = enumerate(data_iter)
        with torch.no_grad():
            for step, batch in step_iter:
                if max_steps is not None and step >= max_steps:
                    break
                args, kwargs = _extract_forward_inputs(batch, extractor)
                cast_args = _cast_to_dtype(args, calibration_dtype)
                cast_kwargs = _cast_to_dtype(kwargs, calibration_dtype)
                with scoped_no_autocast():
                    model(*cast_args, **cast_kwargs)
    finally:
        for hook in rotation_hooks:
            hook.remove()
        for hook in activation_hooks:
            hook.remove()

        rotation_logs: Tuple[str, ...]
        rotation_meta: Dict[str, Dict[str, Any]]
        if enable_rotation:
            rotation_logs, rotation_meta = _apply_rotations(rotation_stats, rotation_mode)
        else:
            rotation_logs, rotation_meta = tuple(), {}

        activation_meta: Dict[str, Dict[str, float]] = {}
        for name, accumulator in activation_accumulators.items():
            stats = accumulator.finalize()
            if stats is not None:
                activation_meta[f"{name}.activation"] = stats

        summary_lines, quant_records = _finalize_model_quantizers(model, activation_meta, rotation_meta)
        summary_records.extend(quant_records)

        for summary in rotation_logs + summary_lines:
            print(summary)
            log_lines.append(summary)

        _write_quant_logs(run_dir, log_lines, summary_records)

        set_observing(model, False)
        set_quant_state(model, weight_quant=False, act_quant=False)
        _restore_tensor_state(model, dtype_snapshot)
        if cfg.get("restore_training", True):
            model.train(training_mode)


def disable_all_quant(model: nn.Module) -> nn.Module:
    """Remove quantization wrappers and restore plain ``nn.Linear`` layers."""

    set_quant_state(model, weight_quant=False, act_quant=False)
    set_observing(model, False)
    set_static_quant(model, static_quant=False)

    for parent, name, child in _walk_named_children(model):
        if isinstance(child, QuantLinear):
            restored = _rebuild_linear(child)
            setattr(parent, name, restored)

    return model


def convert_to_int(
    model: nn.Module,
    *,
    propagate_int: bool = False,
    use_int_kernel: bool = False,
) -> nn.Module:
    """
    Materialise integer weights for quantised wrappers to enable real-int execution.

    When ``propagate_int`` is set the routine also tags downstream wrappers with the
    upstream activation quantiser outputs so alternative runtimes can consume integer
    activations directly.
    """

    _convert_to_int_utils(model, propagate_int=propagate_int, use_int_kernel=use_int_kernel)
    return model


def _walk_named_children(module: nn.Module) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
    for name, child in list(module.named_children()):
        yield module, name, child
        yield from _walk_named_children(child)


def _ensure_triplet_named_modules(module: nn.Module) -> None:
    if getattr(module, "_quant_pct_named_modules_wrapped", False):
        return

    original_named_modules = module.named_modules

    def _named_modules_with_dummy_parent(
        self: nn.Module,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Any]:
        stack = inspect.stack()[1:]
        try:
            triplet_mode = True
            if stack:
                frame_info = stack[0]
                module_name = frame_info.frame.f_globals.get("__name__", "")
                if module_name.startswith("torch.nn") and frame_info.function in {
                    "_named_members",
                    "named_parameters",
                    "named_buffers",
                    "parameters",
                    "buffers",
                    "modules",
                }:
                    triplet_mode = False
        finally:
            del stack
        for name, child in original_named_modules(memo, prefix, remove_duplicate):
            if triplet_mode:
                yield (None, name, child)
            else:
                yield (name, child)

    module._quant_pct_named_modules_original = original_named_modules
    module.named_modules = types.MethodType(_named_modules_with_dummy_parent, module)
    setattr(module, "_quant_pct_named_modules_wrapped", True)

def _extract_forward_inputs(
    batch: _Batch,
    extractor: Optional[_ForwardExtractor],
) -> Tuple[_Args, _Kwargs]:
    if extractor is not None:
        return extractor(batch)
    if isinstance(batch, dict):
        return (), batch
    if isinstance(batch, (list, tuple)):
        return tuple(batch), {}
    return (batch,), {}


def _snapshot_tensor_state(model: nn.Module) -> OrderedDict[str, Tuple[torch.dtype, torch.device]]:
    snapshot: OrderedDict[str, Tuple[torch.dtype, torch.device]] = OrderedDict()
    for name, param in model.named_parameters(recurse=True):
        snapshot[f"param::{name}"] = (param.dtype, param.device)
    for name, buffer in model.named_buffers(recurse=True):
        snapshot[f"buffer::{name}"] = (buffer.dtype, buffer.device)
    return snapshot


def _restore_tensor_state(
    model: nn.Module,
    snapshot: OrderedDict[str, Tuple[torch.dtype, torch.device]],
) -> None:
    for name, param in model.named_parameters(recurse=True):
        key = f"param::{name}"
        if key in snapshot:
            dtype, device = snapshot[key]
            param.data = param.data.to(device=device, dtype=dtype)
    for name, buffer in model.named_buffers(recurse=True):
        key = f"buffer::{name}"
        if key in snapshot:
            dtype, device = snapshot[key]
            buffer.data = buffer.data.to(device=device, dtype=dtype)


def _rebuild_linear(module: QuantLinear) -> nn.Linear:
    has_bias = module.bias is not None
    restored = nn.Linear(module.in_features, module.out_features, bias=has_bias)
    restored = restored.to(device=module.weight.device, dtype=module.weight.dtype)
    with torch.no_grad():
        restored.weight.copy_(module.weight.detach())
        if has_bias and restored.bias is not None:
            restored.bias.copy_(module.bias.detach())
    return restored


def _cast_to_dtype(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor):
        return force_calib_dtype(value, target=dtype)
    if isinstance(value, tuple):
        return tuple(_cast_to_dtype(v, dtype) for v in value)
    if isinstance(value, list):
        return [_cast_to_dtype(v, dtype) for v in value]
    if isinstance(value, OrderedDict):
        return value.__class__((k, _cast_to_dtype(v, dtype)) for k, v in value.items())
    if isinstance(value, dict):
        return {k: _cast_to_dtype(v, dtype) for k, v in value.items()}
    return value


def _write_quant_logs(run_dir: Path, log_lines: list[str], records: list[Dict[str, Any]]) -> None:
    if log_lines:
        with (run_dir / "calibration.log").open("w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
            if log_lines:
                f.write("\n")

    if not records:
        return

    summary_json = run_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    fieldnames = sorted({key for record in records for key in record.keys()})
    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    for record in records:
        layer_slug = record["layer_name"].replace(".", "_")
        filename = f"{layer_slug}_{record['kind']}.json"
        with (run_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)


def _make_rotation_hook(accumulator: "_RotationAccumulator") -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
    def hook(_module: nn.Module, inputs: Tuple[Any, ...], _output: Any) -> None:
        if not inputs:
            return
        accumulator.update(inputs[0])

    return hook


def _make_activation_hook(accumulator: "_ActivationAccumulator") -> Callable[[nn.Module, Tuple[Any, ...]], None]:
    def hook(_module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        if not inputs:
            return
        accumulator.update(inputs[0])

    return hook


def _identify_rotation_targets(model: nn.Module, cfg: Dict[str, Any]) -> Tuple[Tuple[str, QuantLinear], ...]:
    patterns = tuple(p.lower() for p in cfg.get("rotation_patterns", ("gate_proj", "out_proj", "matmul")))
    targets = []
    for name, module in iter_named_modules(model):
        if not isinstance(module, QuantLinear):
            continue
        lname = name.lower()
        if any(pattern in lname for pattern in patterns):
            targets.append((name, module))
    return tuple(targets)


def _apply_rotations(
    rotation_stats: Dict[str, "_RotationAccumulator"],
    rotation_mode: str,
) -> Tuple[Tuple[str, ...], Dict[str, Dict[str, Any]]]:
    logs = []
    metadata: Dict[str, Dict[str, Any]] = {}
    for name, accumulator in rotation_stats.items():
        stats = accumulator.finalize()
        if not stats:
            continue
        module = accumulator.module
        if rotation_mode == "wht+klt":
            rotation = apply_wht_then_klt(stats)
        else:
            rotation = compute_klt_from_stats(stats)

        if rotation is None:
            continue

        with torch.no_grad():
            original_weight = module.weight.detach().clone()
            weight_min_pre = float(original_weight.min().item())
            weight_max_pre = float(original_weight.max().item())
            fold_rotation_into_linear(module, R_in=rotation)
            updated_weight = module.weight.detach()
            delta = torch.linalg.norm((updated_weight - original_weight).reshape(-1))
            weight_min_post = float(updated_weight.min().item())
            weight_max_post = float(updated_weight.max().item())

        if hasattr(module, "weight_quantized"):
            module.weight_quantized = False
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is not None:
            quantizer.scale = None
            quantizer.round_zero_point = None
            quantizer.cached_xmin = None
            quantizer.cached_xmax = None
            if hasattr(quantizer, "observered"):
                quantizer.observered = False

        logs.append(f"[rotation] {name}: mode={rotation_mode}, delta={float(delta):.6g}")
        metadata[name] = {
            "mode": rotation_mode,
            "raw_min": weight_min_pre,
            "raw_max": weight_max_pre,
            "weight_min_pre": weight_min_pre,
            "weight_max_pre": weight_max_pre,
            "weight_min_post": weight_min_post,
            "weight_max_post": weight_max_post,
            "rotation_delta": float(delta),
        }

    return tuple(logs), metadata


class _ActivationAccumulator:
    """Track activation ranges observed during calibration."""

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self._min: Optional[torch.Tensor] = None
        self._max: Optional[torch.Tensor] = None

    def update(self, tensor: Any) -> None:
        if tensor is None:
            return
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if tensor is None:
            return
        data = tensor.detach()
        if data.numel() == 0:
            return
        data_min = data.amin().to(torch.float64)
        data_max = data.amax().to(torch.float64)
        if self._min is None:
            self._min = data_min
            self._max = data_max
        else:
            self._min = torch.minimum(self._min, data_min)
            self._max = torch.maximum(self._max, data_max)

    def finalize(self) -> Optional[Dict[str, float]]:
        if self._min is None or self._max is None:
            return None
        return {
            "raw_min": float(self._min.item()),
            "raw_max": float(self._max.item()),
            "mode": "none",
        }


class _RotationAccumulator:
    """Track second-order statistics for a linear module."""

    def __init__(self, name: str, module: QuantLinear) -> None:
        self.name = name
        self.module = module
        self.count = 0
        self._sum: Optional[torch.Tensor] = None
        self._outer: Optional[torch.Tensor] = None

    def update(self, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if tensor is None:
            return
        x = tensor.detach()
        if x.dim() > 2:
            x = x.flatten(0, -2)
        else:
            x = x.reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        x = x.to(dtype=torch.float64)
        if self._sum is None:
            dim = x.shape[-1]
            device = x.device
            self._sum = torch.zeros(dim, dtype=torch.float64, device=device)
            self._outer = torch.zeros((dim, dim), dtype=torch.float64, device=device)
        self.count += x.shape[0]
        self._sum.add_(x.sum(dim=0))
        self._outer.add_(x.transpose(0, 1) @ x)

    def finalize(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.count == 0 or self._sum is None or self._outer is None:
            return None
        mean = self._sum / self.count
        cov = self._outer / self.count - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.transpose(0, 1))
        return {
            "mean": mean,
            "cov": cov,
            "count": torch.tensor(self.count, dtype=torch.float64, device=cov.device),
        }


def _build_observer_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    activation_name = cfg.get("a_observer", "percentile")
    activation_percent = cfg.get("a_percentile", 0.999)
    weight_name = cfg.get("w_observer", "minmax")
    weight_percent = cfg.get("w_percentile")
    weight_per_channel_axes = cfg.get("w_per_channel_axes", [0])

    activation_cfg = {
        "name": activation_name,
        "percentile": activation_percent,
        "granularity": cfg.get("a_granularity", "tensor"),
        "per_channel_axes": cfg.get("a_per_channel_axes", []),
        "kwargs": cfg.get("a_observer_kwargs", {}),
    }

    weight_cfg = {
        "name": weight_name,
        "percentile": weight_percent,
        "granularity": cfg.get("w_granularity"),
        "per_channel_axes": weight_per_channel_axes,
        "kwargs": cfg.get("w_observer_kwargs", {}),
    }

    return {"activation": activation_cfg, "weight": weight_cfg}


def _configure_model_observers(model: nn.Module, observer_cfg: Dict[str, Dict[str, Any]]) -> None:
    for module in model.modules():
        if isinstance(module, QuantLinear):
            _configure_linear_observers(module, observer_cfg)


def _configure_linear_observers(module: QuantLinear, observer_cfg: Dict[str, Dict[str, Any]]) -> None:
    if hasattr(module, "weight_quantizer") and module.weight_quantizer is not None:
        _configure_quantizer(
            module.weight_quantizer,
            observer_cfg["weight"],
            kind="weight",
        )
    if getattr(module, "act_quantizer", None) is not None:
        _configure_quantizer(
            module.act_quantizer,
            observer_cfg["activation"],
            kind="activation",
        )
    if hasattr(module, "weight_quantized"):
        module.weight_quantized = False


def _configure_quantizer(quantizer, cfg: Dict[str, Any], kind: str) -> None:
    if quantizer is None:
        return

    granularity = cfg.get("granularity")
    if granularity is None:
        granularity = _infer_granularity(quantizer)

    kwargs = dict(cfg.get("kwargs") or {})
    kwargs.setdefault("granularity", granularity)

    percentile = cfg.get("percentile")
    if cfg["name"].lower() == "percentile" and percentile is not None:
        kwargs.setdefault("percent", percentile)

    observer = build_observer(cfg["name"], **kwargs)
    quantizer.observer = observer
    quantizer.is_observing = False
    if hasattr(quantizer, "observered"):
        quantizer.observered = False
    quantizer.cached_xmin = None
    quantizer.cached_xmax = None
    quantizer.scale = None
    quantizer.round_zero_point = None
    quantizer._observer_name = cfg["name"].lower()  # type: ignore[attr-defined]
    quantizer._observer_config = cfg  # type: ignore[attr-defined]


def _infer_granularity(quantizer) -> str:
    axes = getattr(quantizer, "per_channel_axes", []) or []
    if len(axes) == 0:
        return "tensor"
    if len(axes) == 1:
        return f"dim{axes[0]}"
    return [f"dim{axis}" for axis in axes]


def _finalize_model_quantizers(
    model: nn.Module,
    activation_meta: Dict[str, Dict[str, float]],
    rotation_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Tuple[str, ...], List[Dict[str, Any]]]:
    summaries: List[str] = []
    records: List[Dict[str, Any]] = []
    for name, module in iter_named_modules(model):
        if isinstance(module, QuantLinear):
            linear_summaries, linear_records = _finalize_linear_quantizers(
                name,
                module,
                activation_meta,
                rotation_meta,
            )
            summaries.extend(linear_summaries)
            records.extend(linear_records)
    return tuple(summaries), records


def _finalize_linear_quantizers(
    module_name: str,
    module: QuantLinear,
    activation_meta: Dict[str, Dict[str, float]],
    rotation_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Tuple[str, ...], Tuple[Dict[str, Any], ...]]:
    summaries: List[str] = []
    records: List[Dict[str, Any]] = []

    if hasattr(module, "weight_quantizer"):
        rotation_notes = rotation_meta.get(module_name, {})
        log_line, record = _finalize_quantizer(
            qualified_name=f"{module_name}.weight",
            quantizer=module.weight_quantizer,
            kind="weight",
            reference_tensor=getattr(getattr(module, "_origin_linear", None), "weight", None),
            metadata=rotation_notes,
        )
        if log_line:
            summaries.append(log_line)
        if record:
            records.append(record)

    for suffix in ("input_quantizer", "output_quantizer"):
        quantizer = getattr(module, suffix, None)
        if quantizer is None:
            continue
        act_meta = activation_meta.get(f"{module_name}.activation", {})
        log_line, record = _finalize_quantizer(
            qualified_name=f"{module_name}.{suffix}",
            quantizer=quantizer,
            kind="activation",
            reference_tensor=None,
            metadata=act_meta,
        )
        if log_line:
            summaries.append(log_line)
        if record:
            records.append(record)
    return tuple(summaries), records


def _reduce_min(value: Any) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.min().item())
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def _reduce_max(value: Any) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.max().item())
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def _finalize_quantizer(
    qualified_name: str,
    quantizer,
    kind: str,
    reference_tensor: Optional[torch.Tensor],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    observer = getattr(quantizer, "observer", None)
    observer_name = getattr(quantizer, "_observer_name", "unknown")
    percentile = None
    if observer is None:
        cached_min = _reduce_min(getattr(quantizer, "cached_xmin", None))
        cached_max = _reduce_max(getattr(quantizer, "cached_xmax", None))
        scale_min = _reduce_min(getattr(quantizer, "scale", None))
        scale_max = _reduce_max(getattr(quantizer, "scale", None))
        zero_attr = getattr(quantizer, "round_zero_point", None)
        if zero_attr is None:
            zero_attr = getattr(quantizer, "zero_point", None)
        zero_min = _reduce_min(zero_attr)
        zero_max = _reduce_max(zero_attr)
        zero_desc = "None"
        if zero_min is not None and zero_max is not None:
            zero_desc = f"[{zero_min:.6g}, {zero_max:.6g}]"
        raw_percent = getattr(quantizer, "percent", None)
        if raw_percent is None:
            raw_percent = getattr(quantizer, "percentile", None)
        if isinstance(raw_percent, numbers.Number):
            percentile_desc = f"{float(raw_percent):.6g}"
        else:
            percentile_desc = None
        if scale_min is not None and scale_max is not None:
            clip_min_val = cached_min if cached_min is not None else -abs(scale_min)
            clip_max_val = cached_max if cached_max is not None else abs(scale_max)
            record = {
                "layer_name": qualified_name,
                "kind": kind,
                "observer": observer_name,
                "percentile": percentile_desc,
                "clip_min": clip_min_val,
                "clip_max": clip_max_val,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "zero_point_min": zero_min,
                "zero_point_max": zero_max,
                "fold_mode": metadata.get("mode", "none") if metadata else "none",
                "notes": metadata.get("notes", "") if metadata else "observer_already_frozen",
            }
            if metadata:
                record.update(metadata)
                if "mode" in record:
                    record["fold_mode"] = record.pop("mode")
            log_line = (
                f"[calibrate] {qualified_name}: observer={observer_name}"
                + (f", percentile={percentile_desc}" if percentile_desc else "")
                + f", clip=({clip_min_val:.6g}, {clip_max_val:.6g}), scale=[{scale_min:.6g}, {scale_max:.6g}], zero_point={zero_desc}"
            )
            return log_line, record
        record = {
            "layer_name": qualified_name,
            "kind": kind,
            "observer": observer_name,
            "percentile": None,
            "clip_min": None,
            "clip_max": None,
            "scale_min": None,
            "scale_max": None,
            "zero_point_min": None,
            "zero_point_max": None,
            "fold_mode": metadata.get("mode", "none") if metadata else "none",
            "notes": "observer_already_frozen",
        }
        if metadata:
            record.update(metadata)
            if "mode" in record:
                record["fold_mode"] = record.pop("mode")
        return f"[calibrate] {qualified_name}: observer={observer_name} already frozen", record

    if getattr(observer, "left_percent", None) is not None:
        percentile = getattr(observer, "left_percent")
    elif getattr(observer, "percentile", None) is not None:
        percentile = getattr(observer, "percentile")

    with torch.no_grad():
        if reference_tensor is not None:
            observer.update(reference_tensor)
        xmin, xmax = observer.cal_min_max()

    if quantizer.symmetric or quantizer.disable_zero_point:
        quantizer.symmetric_cal_scale(xmin, xmax)
    else:
        quantizer.assymmetric_cal_scale(xmin, xmax)

    if reference_tensor is not None:
        quantizer.scale = quantizer.expand_scale_shape_2_x(reference_tensor, quantizer.scale)
        if quantizer.round_zero_point is not None:
            quantizer.round_zero_point = quantizer.expand_scale_shape_2_x(
                reference_tensor, quantizer.round_zero_point
            )

    quantizer.cached_xmin = xmin
    quantizer.cached_xmax = xmax
    quantizer.is_observing = False
    quantizer.observer = None
    if hasattr(quantizer, "observered"):
        quantizer.observered = True

    clip_min = float(xmin.min().item()) if xmin.numel() else float(xmin)
    clip_max = float(xmax.max().item()) if xmax.numel() else float(xmax)
    scale = quantizer.scale
    scale_min = float(scale.min().item()) if isinstance(scale, torch.Tensor) else float(scale)
    scale_max = float(scale.max().item()) if isinstance(scale, torch.Tensor) else float(scale)
    zero = quantizer.round_zero_point
    if isinstance(zero, torch.Tensor):
        zero_min = float(zero.min().item())
        zero_max = float(zero.max().item())
        zero_desc = f"[{zero_min:.6g}, {zero_max:.6g}]"
    else:
        zero_min = zero_max = None
        zero_desc = "None"

    percentile_desc = None
    if percentile is not None:
        if isinstance(percentile, (list, tuple)):
            percentile_desc = ", ".join(f"{p:.6g}" for p in percentile)
        else:
            percentile_desc = f"{percentile:.6g}"

    record: Dict[str, Any] = {
        "layer_name": qualified_name,
        "kind": kind,
        "observer": observer_name,
        "percentile": percentile_desc,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "zero_point_min": zero_min,
        "zero_point_max": zero_max,
        "fold_mode": metadata.get("mode", "none") if metadata else "none",
        "notes": metadata.get("notes", "") if metadata else "",
    }

    if metadata:
        record.update(metadata)
        if "mode" in record:
            record["fold_mode"] = record.pop("mode")

    if reference_tensor is not None:
        record.setdefault("weight_min_post", float(reference_tensor.min().item()))
        record.setdefault("weight_max_post", float(reference_tensor.max().item()))

    if metadata and "raw_min" in metadata and "raw_max" in metadata:
        record.setdefault("raw_min", metadata["raw_min"])
        record.setdefault("raw_max", metadata["raw_max"])
    elif reference_tensor is not None and kind == "weight":
        record.setdefault("raw_min", float(reference_tensor.min().item()))
        record.setdefault("raw_max", float(reference_tensor.max().item()))

    log_line = (
        f"[calibrate] {qualified_name}: observer={observer_name}"
        + (f", percentile={percentile_desc}" if percentile_desc else "")
        + f", clip=({clip_min:.6g}, {clip_max:.6g}), scale=[{scale_min:.6g}, {scale_max:.6g}], zero_point={zero_desc}"
    )

    return log_line, record

# Dispatcher ------------------------------------------------------------------------------------


def calibrate(model, *args, **kwargs):
    """Dispatch to percentile or quantization calibration based on provided arguments."""
    if len(args) >= 2:
        dataloader, cfg = args[0], args[1]
        if isinstance(dataloader, DataLoader) and isinstance(cfg, QuantConfig):
            remaining_args = args[2:]
            return calibrate_percentiles(model, dataloader, cfg, *remaining_args, **kwargs)

    if "dataloader" in kwargs and "cfg" in kwargs:
        dataloader = kwargs["dataloader"]
        cfg = kwargs["cfg"]
        if isinstance(dataloader, DataLoader) and isinstance(cfg, QuantConfig):
            remaining_kwargs = dict(kwargs)
            remaining_kwargs.pop("dataloader")
            remaining_kwargs.pop("cfg")
            return calibrate_percentiles(model, dataloader, cfg, *args, **remaining_kwargs)

    return calibrate_quantization(model, *args, **kwargs)

