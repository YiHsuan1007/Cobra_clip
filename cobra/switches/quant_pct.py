"""Runtime control helpers for percentile clipping and quantization toggles."""
from __future__ import annotations

import csv
import json
import logging
import types
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import inspect
import torch
from torch import nn
from torch.utils.data import DataLoader

from cobra.integration.hooks import (
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
from cobra.quantize.utils import (
    convert_to_int as _convert_to_int_utils,
    enable_observation as _enable_observation_utils,
    finalize_all_quantizers as _finalize_all_quantizers,
    iter_named_modules,
    set_observing,
    set_quant_state,
    set_static_quant,
)
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
]

_HANDLE_ATTR = "_quant_pct_handles"
_OBSERVER_ATTR = "_quant_pct_observers"

ActivationCallback = Callable[[str, str, object], None]

DEFAULT_TARGETS: Sequence[str] = tuple(DEFAULT_PERCENTILE_TARGETS)
LEGACY_TARGET_MAP = {
    "vision.dino": "dino",
    "vision.siglip": "siglip",
    "mm.out": "fused",
}

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


def _should_wrap_name(name: str, prefixes: Optional[Sequence[str]]) -> bool:
    if not prefixes:
        return True
    return any(name.startswith(prefix) for prefix in prefixes)


def _log_replacement_summary(
    kind: str,
    names: Sequence[str],
    max_expected: Optional[int],
    raise_on_excess: bool,
) -> None:
    count = len(names)
    preview = ", ".join(names[:10])
    suffix = f": {preview}" if preview else ""
    logging.info("[QuantPct][%s] Replaced %d module(s)%s", kind, count, suffix)
    if max_expected is None:
        return
    if count > max_expected:
        message = (
            f"[QuantPct][{kind}] Replacement count {count} exceeds calibrated target budget {max_expected}."
        )
        if raise_on_excess:
            raise RuntimeError(message)
        logging.warning(message)


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


# Percentile clipping helpers -------------------------------------------------------------------

def calibrate_percentiles(
    model,
    dataloader: DataLoader,
    cfg: QuantConfig,
    targets: Optional[Iterable[str]] = None,
) -> dict:
    """Run calibration and persist observer statistics."""
    return calibrate_model(model, dataloader, cfg, targets=targets)


def activate_observers(model: nn.Module) -> None:
    """Enable observation on quantization wrappers."""
    _enable_observation_utils(model)


def finalize_quant_params(model: nn.Module) -> None:
    """Finalize quantizer parameters prior to real-int execution."""
    _finalize_all_quantizers(model)


def _build_observer(state: dict, cfg: QuantConfig, target: str) -> PercentileObserver:
    observer = PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples, target=target)
    observer.load_state_dict(state)
    return observer


def _collect_observers(
    stats: dict,
    cfg: QuantConfig,
) -> Dict[str, PercentileObserver]:
    observers = stats.get("observers", {})
    registered_targets = stats.get("targets") or DEFAULT_TARGETS
    result: Dict[str, PercentileObserver] = {}
    for name in registered_targets:
        state = observers.get(name)
        if state is None and name in LEGACY_TARGET_MAP:
            state = observers.get(LEGACY_TARGET_MAP[name])
        if state is None:
            raise KeyError(f"Calibration statistics are missing required observer entries for `{name}`.")
        result[name] = _build_observer(state, cfg, target=name)
    return result


def enable(
    model,
    cfg: QuantConfig,
    *,
    mode: str = "apply",
    dumper: Optional[ActivationCallback] = None,
    targets: Optional[Iterable[str]] = None,
) -> None:
    """Enable percentile clipping by registering forward hooks on ``model``."""

    if targets is not None:
        selected_targets = tuple(targets)
    elif cfg.targets:
        selected_targets = tuple(cfg.targets)
    else:
        selected_targets = tuple(DEFAULT_TARGETS)

    normalized_mode = mode.lower()
    if normalized_mode not in {"apply", "collect", "off"}:
        raise ValueError("`mode` must be one of {\"apply\", \"collect\", \"off\"}.")

    if normalized_mode == "off":
        disable(model)
        return

    disable(model)

    observer_map: Dict[str, PercentileObserver]
    if normalized_mode == "collect":
        observer_map = {
            name: PercentileObserver(cfg.p_max, cfg.mode, cfg.max_samples, target=name)
            for name in selected_targets
        }
        handles = attach_percentile_hooks(
            model,
            observers=observer_map,
            apply_clipping=False,
            targets=selected_targets,
            dumper=dumper,
        )
    else:  # apply
        stats = load_stats(cfg.stats_path)
        cfg_from_stats = stats.get("config", {})
        cfg.p_max = cfg_from_stats.get("p_max", cfg.p_max)
        cfg.mode = cfg_from_stats.get("mode", cfg.mode)
        cfg.max_samples = cfg_from_stats.get("max_samples", cfg.max_samples)

        observer_map_all = _collect_observers(stats, cfg)
        try:
            observer_map = {name: observer_map_all[name] for name in selected_targets}
        except KeyError as exc:
            missing = exc.args[0]
            raise KeyError(f"Observer statistics for target `{missing}` are unavailable in `{cfg.stats_path}`.") from exc
        handles = attach_percentile_hooks(
            model,
            observers=observer_map,
            apply_clipping=True,
            targets=selected_targets,
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
        quant_layer._origin_linear = base_module  # type: ignore[attr-defined]

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
    return quant_layer


def _is_quant_wrapped(module: nn.Module) -> bool:
    wrapper_cls = globals().get("QuantMatmulWrapper")
    if wrapper_cls is not None and isinstance(module, wrapper_cls):
        return True
    return bool(getattr(module, "_quant_wrapped", False))


def _mark_quant_wrapped(module: nn.Module) -> None:
    setattr(module, "_quant_wrapped", True)


def _is_matmul_module(module: nn.Module) -> bool:
    wrapper_cls = globals().get("QuantMatmulWrapper")
    if _is_quant_wrapped(module) or (wrapper_cls is not None and isinstance(module, wrapper_cls)):
        return False
    if isinstance(module, QuantMatMul):
        return False
    if module.__class__.__name__.lower() == "matmul":
        return True
    return getattr(module, "torch_fn", None) is torch.matmul


def replace_linear_layers(
    module: nn.Module,
    cfg,
    *,
    weight_bits: Optional[int] = None,
    act_bits: Optional[int] = None,
    targets: Optional[Sequence[str]] = None,
    max_expected: Optional[int] = None,
    raise_on_excess: bool = False,
) -> List[str]:
    """
    Recursively replace nn.Linear modules with QuantLinear instances.

    The helper expects ``cfg`` to expose ``weight_bits`` / ``act_bits`` attributes, but callers
    can override the values explicitly via the keyword arguments.
    """

    resolved_weight_bits = int(
        weight_bits if weight_bits is not None else getattr(cfg, "weight_bits", getattr(cfg, "wbits", 8))
    )
    resolved_act_bits = int(
        act_bits if act_bits is not None else getattr(cfg, "act_bits", getattr(cfg, "abits", 8))
    )

    cfg_targets = _get_cfg_value(cfg, "targets")
    prefixes = _normalize_target_prefixes(targets if targets is not None else cfg_targets)
    if prefixes is not None and max_expected is None:
        max_expected = len(prefixes)
    strict_budget = raise_on_excess or bool(_get_cfg_value(cfg, "strict_target_budget", False))

    replacements: List[str] = []

    def _walk(current: nn.Module, prefix: str) -> None:
        for child_name, child in list(current.named_children()):
            if isinstance(child, QuantLinear):
                continue
            fq_name = f"{prefix}.{child_name}" if prefix else child_name
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
    _log_replacement_summary("Linear", replacements, max_expected, strict_budget)
    return replacements

# 10/27新增：替換其他層的量化包裝版本


def _collect_conv_targets(root: nn.Module) -> List[Tuple[nn.Module, str, nn.Conv2d, str]]:
    """Depth-first search for unwrapped Conv2d modules."""
    targets: List[Tuple[nn.Module, str, nn.Conv2d, str]] = []
    visited: Set[int] = set()

    def _dfs(module: nn.Module, path: str) -> None:
        module_id = id(module)
        if module_id in visited:  # visited prevents revisiting the same module through different parents.
            return
        visited.add(module_id)

        for name, child in module.named_children():
            if getattr(child, "_quant_pct_wrapped", False):  # wrapped flag short-circuits newly created quant layers.
                continue
            if isinstance(child, QuantConvBase):
                continue
            fq_name = f"{path}.{name}" if path else name
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
    targets: Optional[Sequence[str]] = None,
    max_expected: Optional[int] = None,
    raise_on_excess: bool = False,
) -> List[str]:
    """Replace convolution modules with quantized variants."""

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
    prefixes = _normalize_target_prefixes(targets if targets is not None else cfg_targets)
    if prefixes is not None and max_expected is None:
        max_expected = len(prefixes)
    strict_budget = raise_on_excess or bool(_get_cfg_value(cfg, "strict_target_budget", False))

    replacements: List[str] = []
    conv_targets = _collect_conv_targets(module)
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

    _ensure_triplet_named_modules(module)
    _log_replacement_summary("Conv", replacements, max_expected, strict_budget)
    return replacements


def replace_matmul_layers(
    module: nn.Module,
    cfg: Any,
    *,
    act_bits: Optional[int] = None,
    disable_act_quant: Optional[bool] = None,
    observe: Optional[str] = None,
    visited: Optional[Set[int]] = None,
    targets: Optional[Sequence[str]] = None,
    max_expected: Optional[int] = None,
    raise_on_excess: bool = False,
) -> List[str]:
    """Recursively replace MatMul helpers with ``QuantMatMul``."""

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

    if visited is None:
        visited = set()

    cfg_targets = _get_cfg_value(cfg, "targets")
    prefixes = _normalize_target_prefixes(targets if targets is not None else cfg_targets)
    if prefixes is not None and max_expected is None:
        max_expected = len(prefixes)
    strict_budget = raise_on_excess or bool(_get_cfg_value(cfg, "strict_target_budget", False))

    targets_list: List[Tuple[nn.Module, str, nn.Module, str]] = []

    def _dfs(current: nn.Module, path: str) -> None:
        module_id = id(current)
        if module_id in visited:
            return
        visited.add(module_id)

        for name, child in list(current.named_children()):
            if _is_quant_wrapped(child):
                continue
            if isinstance(child, QuantMatMul):
                continue
            fq_name = f"{path}.{name}" if path else name
            if _is_matmul_module(child):
                if _should_wrap_name(fq_name, prefixes):
                    targets_list.append((current, name, child, fq_name))
                    continue
                _dfs(child, fq_name)
                continue
            _dfs(child, fq_name)

    _dfs(module, "")

    replacements: List[str] = []
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
        replacements.append(fq_name)
    _log_replacement_summary("MatMul", replacements, max_expected, strict_budget)
    return replacements

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

