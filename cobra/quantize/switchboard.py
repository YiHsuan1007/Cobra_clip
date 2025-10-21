"""High level helpers to toggle quantization on cobra models.

This mirrors the workflow used in MambaQuant projects while keeping the
module hierarchy under ``cobra.quantize``.
"""

from __future__ import annotations

from collections import OrderedDict
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch import nn

from .int_linear import QuantLinear
from .observers import get_observer
from .rotations import apply_wht_then_klt, compute_klt_from_stats, fold_rotation_into_linear
from .utils import set_observing, set_quant_state, set_static_quant
from .utils.dtype import force_calib_dtype, scoped_no_autocast

_Batch = Any
_Args = Tuple[Any, ...]
_Kwargs = Dict[str, Any]
_ForwardExtractor = Callable[[Any], Tuple[_Args, _Kwargs]]


def enable_quant(model: nn.Module, **cfg: Any) -> nn.Module:
    """Replace supported layers with their quantized counterparts.

    Parameters
    ----------
    model:
        Target module whose ``nn.Linear`` layers should be wrapped.
    **cfg:
        Optional configuration such as ``weight_quant_params`` or
        ``act_quant_params``. Any missing configuration falls back to the
        defaults embedded inside ``QuantLinear``.
    """

    weight_quant_params = dict(cfg.get("weight_quant_params", {"dynamic_method": "per_tensor"}))
    act_quant_params = dict(cfg.get("act_quant_params", {"dynamic_method": "per_tensor"}))
    disable_input_quant = cfg.get("disable_input_quant", False)

    observer_cfg = _build_observer_config(cfg)

    if observer_cfg["weight"].get("per_channel_axes") is not None:
        weight_quant_params.setdefault("per_channel_axes", observer_cfg["weight"]["per_channel_axes"])
    act_quant_params.setdefault("per_channel_axes", observer_cfg["activation"].get("per_channel_axes", []))

    observe_token = observer_cfg["activation"]["name"]

    for parent, name, child in _walk_named_children(model):
        if isinstance(child, nn.Linear) and not isinstance(child, QuantLinear):
            quant_layer = QuantLinear(
                child,
                weight_quant_params=weight_quant_params,
                act_quant_params=act_quant_params,
                disable_input_quant=disable_input_quant,
                observe=observe_token,
            )
            quant_layer._origin_linear = child  # type: ignore[attr-defined]
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


def calibrate(
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
    for name, module in model.named_modules():
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


def _walk_named_children(module: nn.Module) -> Iterator[Tuple[nn.Module, str, nn.Module]]:
    for name, child in list(module.named_children()):
        yield module, name, child
        yield from _walk_named_children(child)


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
    for name, module in model.named_modules():
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

    observer = get_observer(cfg["name"], **kwargs)
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
    summaries: list[str] = []
    records: list[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            lines, layer_records = _finalize_linear_quantizers(
                name,
                module,
                activation_meta,
                rotation_meta,
            )
            summaries.extend(lines)
            records.extend(layer_records)
    return tuple(summaries), records


def _finalize_linear_quantizers(
    name: str,
    module: QuantLinear,
    activation_meta: Dict[str, Dict[str, float]],
    rotation_meta: Dict[str, Dict[str, Any]],
) -> Tuple[Tuple[str, ...], List[Dict[str, Any]]]:
    summaries: list[str] = []
    records: list[Dict[str, Any]] = []
    base_meta = rotation_meta.get(name, {})
    if hasattr(module, "weight_quantizer") and module.weight_quantizer is not None:
        log_line, record = _finalize_quantizer(
            f"{name}.weight",
            module.weight_quantizer,
            kind="weight",
            reference_tensor=module.weight,
            metadata=base_meta,
        )
        if log_line:
            summaries.append(log_line)
        if record:
            records.append(record)
    if getattr(module, "act_quantizer", None) is not None:
        act_meta = activation_meta.get(f"{name}.activation", {})
        log_line, record = _finalize_quantizer(
            f"{name}.activation",
            module.act_quantizer,
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
