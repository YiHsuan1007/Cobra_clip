"""CLI entrypoint to enable percentile clipping with low-bit linear quantization."""
from __future__ import annotations

import argparse
import json
import logging
import math
import numbers
import os
import time
import warnings
from pathlib import Path
from collections import defaultdict
import types
from difflib import get_close_matches
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cobra import load as load_model
from cobra.quantize.calibrate import (
    _cast_float_payload,
    _extract_text_inputs,
    _move_to_device,
    load_stats,
)
from cobra.quantize.config import QuantConfig
from cobra.quantize.quantizer import UniformAffineQuantizer
from cobra.quantize.int_linear import QuantLinear
from cobra.quantize.int_conv import QuantConvBase
from cobra.quantize.int_matmul import QuantMatMul
from cobra.switches import quant_pct
from cobra.quantize.utils import (
    assert_all_initialized,
    convert_to_int,
    count_uninitialized_quantizers,
    enable_observation,
    count_observers,
    finalize_all_quantizers,
    freeze_weight_qparams,
    iter_quantizers,
    register_scales_and_zeros,
    set_quant_state,
    set_static_quant,
    set_observing,
    summarize_quantizer_init,
)
from cobra.quantize.percentile_aliases import normalize_targets, expand_targets_for_hooks, normalize_target_name
from cobra.quantize.utils.percentile_to_overrides import (
    apply_overrides as _apply_percentile_overrides,
    build_percentile_overrides,
    load_stats as _load_percentile_stats,
    save_overrides as _save_percentile_overrides,
)
from cobra.utils.latency_meter import LatencyMeter
from cobra.utils.mem_peak import init_peak_track

from cobra.calibrate.enable_pct import (
    ActivationDumper,
    _DEFAULT_DUMP_POINTS,
    _extract_clip_values,
    _parse_dump_targets,
    _parse_targets,
    _register_passthrough_hooks,
    _emit_mem_peak,
)

if hasattr(quant_pct, "canonical_quant_key"):
    _canonical_quant_key = quant_pct.canonical_quant_key
else:
    def _canonical_quant_key(module_path: str, role: str) -> str:
        role_token = role
        if role_token not in ("weight_quantizer", "act_quantizer"):
            role_lower = str(role_token).lower()
            if role_lower.startswith("weight"):
                role_token = "weight_quantizer"
            else:
                role_token = "act_quantizer"
        cleaned = module_path.strip(".")
        return f"{cleaned}.{role_token}"

if hasattr(quant_pct, "rewrite_percentile_module_path"):
    _rewrite_module_path = quant_pct.rewrite_percentile_module_path
else:
    def _rewrite_module_path(path: str) -> str:
        return path


_CALIB_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CALIB_SOURCE_GUIDANCE = (
    "Provide calibration images via CLI (--calib-data /path/to/images), "
    "QuantConfig fields (calibration_data|calib_data|calibration_root|data_root), "
    "or environment (export COBRA_CALIB_DATA=/path/to/images). Examples:\n"
    "  export COBRA_CALIB_DATA=/work/calib_images\n"
    "  python -m cobra.calibrate.enable_pct_lowbit --ckpt CKPT --cfg CONFIG "
    "--real-quant --calib-data /work/calib_images"
)

_DEFAULT_TARGET_PREFIXES: Tuple[str, ...] = ("vision.dino", "vision.siglip", "mm.out")
_DEFAULT_CANONICAL_TARGETS: Tuple[str, ...] = ("vision_backbone", "llm_backbone", "projector")
_DEFAULT_OVERRIDES_PATH = Path("outputs/best_percentile_map.pt")

logger = logging.getLogger(__name__)

if hasattr(quant_pct, "activate_observers"):
    _activate_observers = quant_pct.activate_observers
else:
    _activate_observers = enable_observation

if hasattr(quant_pct, "finalize_quant_params"):
    _finalize_quant_params = quant_pct.finalize_quant_params
else:
    _finalize_quant_params = finalize_all_quantizers


class _CalibrationDataset(Dataset):
    """Image dataset mirroring the percentile calibration loader."""

    def __init__(self, paths: Sequence[Path], transform: Callable[[Image.Image], object]) -> None:
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("Calibration dataset is empty.")
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        path = self.paths[index]
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values}


def _collate_calibration(batch: List[Dict[str, object]]) -> Dict[str, object]:
    pixel_values = [item["pixel_values"] for item in batch]
    first = pixel_values[0]
    if isinstance(first, dict):
        return {
            "pixel_values": {key: torch.stack([pv[key] for pv in pixel_values], dim=0) for key in first}
        }
    return {"pixel_values": torch.stack(pixel_values, dim=0)}


def _reduce_min(value: object) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.min().item())
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def _reduce_max(value: object) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.max().item())
    if isinstance(value, numbers.Number):
        return float(value)
    return None


def _emit_calibration_logs(model: nn.Module) -> int:
    """Emit `[calibrate] ... scale=[...]` summaries for initialized quantizers."""
    emitted = 0
    for handle in iter_quantizers(model):
        quantizer = handle.quantizer
        scale = getattr(quantizer, "scale", None)
        if isinstance(scale, torch.Tensor):
            if scale.numel() == 0:
                continue
            scale_min = float(scale.min().item())
            scale_max = float(scale.max().item())
        elif isinstance(scale, numbers.Number):
            scale_min = scale_max = float(scale)
        else:
            continue

        clip_min = _reduce_min(getattr(quantizer, "cached_xmin", None))
        clip_max = _reduce_max(getattr(quantizer, "cached_xmax", None))
        if clip_min is None:
            clip_min = -abs(scale_min)
        if clip_max is None:
            clip_max = abs(scale_max)

        zero_attr = getattr(quantizer, "round_zero_point", None)
        if zero_attr is None:
            zero_attr = getattr(quantizer, "zero_point", None)
        zero_min = _reduce_min(zero_attr)
        zero_max = _reduce_max(zero_attr)
        zero_desc = "None" if zero_min is None or zero_max is None else f"[{zero_min:.6g}, {zero_max:.6g}]"

        observer_name = getattr(quantizer, "_observer_name", "unknown")
        raw_percent = getattr(quantizer, "percent", getattr(quantizer, "percentile", None))
        if isinstance(raw_percent, numbers.Number):
            percentile_desc = f"{float(raw_percent):.6g}"
        else:
            percentile_desc = None

        log_line = (
            f"[calibrate] {handle.label}: observer={observer_name}"
            + (f", percentile={percentile_desc}" if percentile_desc else "")
            + f", clip=({clip_min:.6g}, {clip_max:.6g}), scale=[{scale_min:.6g}, {scale_max:.6g}], zero_point={zero_desc}"
        )
        print(log_line)
        emitted += 1

    if emitted == 0:
        print("[calibrate] <no_quantizers>: observer=none, clip=(0, 0), scale=[0, 0], zero_point=None")
    return emitted


def _export_int_weights(model: nn.Module, export_dir: Path) -> Optional[Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict[str, torch.Tensor]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, (QuantLinear, QuantConvBase, QuantMatMul)):
            continue
        weight_int = getattr(module, "weight_int", None)
        if not isinstance(weight_int, torch.Tensor) or weight_int.numel() == 0:
            continue
        entry: Dict[str, torch.Tensor] = {"weight_int": weight_int.detach().cpu()}
        for attr in ("w_scale", "w_zero"):
            tensor = getattr(module, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                entry[attr] = tensor.detach().cpu()
        payload[module_name] = entry
    if not payload:
        logger.info("[export] No INT weights detected; skipping export to %s", export_dir)
        return None
    target = export_dir / "int8_weights.pt"
    torch.save(payload, target)
    return target


def _summarize_override_entries(
    model: nn.Module,
    overrides: Mapping[str, Mapping[str, object]],
    limit: int = 10,
) -> Tuple[int, int, List[Tuple[str, float]]]:
    named_modules = dict(model.named_modules())
    preview: List[Tuple[str, float]] = []
    linear_count = 0
    conv_count = 0

    for override_key, payload in overrides.items():
        if not isinstance(override_key, str) or "." not in override_key:
            continue
        module_name, attr = override_key.rsplit(".", 1)
        module = named_modules.get(module_name)
        if module is None or not isinstance(payload, Mapping):
            continue
        percentile_value = payload.get("percentile", payload.get("percent"))
        if percentile_value is None:
            continue
        try:
            percentile_float = float(percentile_value)
        except (TypeError, ValueError):
            continue
        if isinstance(module, QuantLinear):
            linear_count += 1
        elif isinstance(module, QuantConvBase):
            conv_count += 1
        else:
            continue
        if len(preview) < limit:
            preview.append((f"{module_name}.{attr}", percentile_float))
    return linear_count, conv_count, preview


def _emit_override_report(
    model: nn.Module,
    overrides: Mapping[str, Mapping[str, object]],
    applied_count: int,
    label: str = "PercentileOverrides",
) -> None:
    if applied_count <= 0 or not overrides:
        return
    linear_count, conv_count, preview = _summarize_override_entries(model, overrides)
    if preview:
        print(f"[{label}] sample overrides:")
        for module_name, percentile in preview:
            print(f"  - {module_name} -> percentile=p{percentile:.4g}")
    print(f"[QuantPct][apply] overrides_applied={applied_count} linear={linear_count} conv={conv_count}")


def _normalize_apply_targets(user_targets: Any) -> str:
    if isinstance(user_targets, (list, tuple, set)):
        joined = ",".join(str(entry) for entry in user_targets)
    else:
        joined = str(user_targets or "")
    raw_entries = [entry.strip() for entry in joined.split(",") if entry.strip()]
    alias_map = {
        "vision.dino": ["vision_backbone"],
        "vision.siglip": ["vision_backbone"],
        "mm.out": ["projector"],
        "ssm": ["llm_backbone"],
    }
    expanded: List[str] = []
    for entry in raw_entries:
        expanded.extend(alias_map.get(entry, [entry]))
    seen: Set[str] = set()
    normalized: List[str] = []
    for entry in expanded:
        if entry not in seen:
            seen.add(entry)
            normalized.append(entry)
    return ",".join(normalized)


def _count_prefixed_keys(mapping: Mapping[str, Any], prefix: str) -> int:
    return sum(1 for key in mapping if isinstance(key, str) and key.startswith(prefix))


def _summarize_stats_payload(stats: Mapping[str, Any]) -> Dict[str, int]:
    summary = {
        "root_observer": _count_prefixed_keys(stats, "observer::"),
        "root_target": _count_prefixed_keys(stats, "target::"),
        "observer_entries": 0,
        "canonical_targets": 0,
    }
    observers_block = stats.get("observers")
    if isinstance(observers_block, Mapping):
        summary["observer_entries"] = len(observers_block)
    targets_block = stats.get("targets")
    if isinstance(targets_block, Sequence) and not isinstance(targets_block, (str, bytes)):
        summary["canonical_targets"] = len(tuple(targets_block))
    return summary


def _log_stats_summary(stats: Mapping[str, Any], *, stage: str, warn_if_no_observers: bool = False) -> Dict[str, int]:
    summary = _summarize_stats_payload(stats)
    print(
        f"[QuantPct][{stage}] stats summary: "
        f"root_observer={summary['root_observer']} "
        f"root_target={summary['root_target']} "
        f"observers_map={summary['observer_entries']} "
        f"canonical_targets={summary['canonical_targets']}"
    )
    if warn_if_no_observers and summary["root_observer"] == 0:
        warnings.warn(
            "[QuantPct][collect] Stats file only contains target:: entries. "
            "Apply may report missing observers. Consider removing --dump-where or using the default collect configuration.",
            RuntimeWarning,
        )
    return summary


def _log_stats_summary_from_path(path: Path, *, stage: str, warn_if_no_observers: bool = False) -> Optional[Dict[str, Any]]:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        print(f"[QuantPct][{stage}] Stats file {resolved} not found; skipping summary.")
        return None
    try:
        stats = load_stats(resolved)
    except Exception as exc:
        print(f"[QuantPct][{stage}] Failed to load stats from {resolved}: {exc}")
        return None
    if not isinstance(stats, dict):
        print(f"[QuantPct][{stage}] Stats payload at {resolved} is not a mapping (type={type(stats).__name__}).")
        return None
    _log_stats_summary(stats, stage=stage, warn_if_no_observers=warn_if_no_observers)
    return stats


def _infer_synthetic_image_hw(vision: object, transform: object) -> Tuple[int, int]:
    def _normalize(candidate: object) -> Optional[Tuple[int, int]]:
        if candidate is None:
            return None
        if isinstance(candidate, (tuple, list)):
            numeric = [int(x) for x in candidate if isinstance(x, (int, float))]
            if len(numeric) >= 2:
                return max(1, numeric[-2]), max(1, numeric[-1])
            if len(numeric) == 1:
                value = max(1, numeric[0])
                return value, value
            return None
        if isinstance(candidate, (int, float)):
            value = max(1, int(candidate))
            return value, value
        return None

    for source in (vision, transform):
        if source is None:
            continue
        for attr in ("image_size", "img_size", "input_size", "crop_size", "size"):
            try:
                value = getattr(source, attr)
            except AttributeError:
                continue
            hw = _normalize(value)
            if hw is not None:
                return hw
    return (384, 384)


class _TinySyntheticCalibrationDataset(Dataset):
    """Synthetic image dataset used when no calibration data is available."""

    def __init__(self, transform: Optional[Callable[[Image.Image], object]], image_hw: Tuple[int, int], length: int) -> None:
        self.transform = transform
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.length = max(1, int(length))

    def __len__(self) -> int:  # type: ignore[override]
        return self.length

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        height, width = self.image_hw
        pixels = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8)
        image = Image.fromarray(pixels.numpy(), mode="RGB")
        if self.transform is not None:
            pixel_values = self.transform(image)
        else:
            pixel_values = pixels.permute(2, 0, 1).float().div(255.0)
        return {"pixel_values": pixel_values}


def _build_tiny_synthetic_loader(
    model: nn.Module,
    cfg: QuantConfig,
    device: torch.device,
    length: int = 2,
) -> DataLoader:
    vision = getattr(model, "vision_backbone", None)
    transform = getattr(vision, "image_transform", None) if vision is not None else None
    image_hw = _infer_synthetic_image_hw(vision, transform)
    dataset = _TinySyntheticCalibrationDataset(transform, image_hw, length)
    batch_size = max(1, min(cfg.batch_size, length))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_calibration,
        drop_last=False,
    )


def _iter_uniform_quantizers(candidate: Any) -> List[UniformAffineQuantizer]:
    quantizers: List[UniformAffineQuantizer] = []
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
        if isinstance(obj, nn.ModuleDict):
            for value in obj.values():
                _collect(value)
            return
        if isinstance(obj, (nn.ModuleList, nn.Sequential)):
            for value in obj:
                _collect(value)

    _collect(candidate)
    return quantizers


def _iter_named_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    for entry in model.named_modules():
        if isinstance(entry, tuple):
            if len(entry) < 2:
                continue
            module_name, module = entry[0], entry[1]
        else:
            module_name, module = getattr(entry, "name", ""), entry
        if not isinstance(module, nn.Module):
            continue
        yield module_name, module


def _resolve_calib_root(
    args: argparse.Namespace,
    cfg: QuantConfig,
    env: Mapping[str, str] | None = None,
) -> tuple[Optional[Path], Optional[str]]:
    env_map = os.environ if env is None else env

    cli_value = getattr(args, "calib_data", None)
    if cli_value:
        return Path(cli_value).expanduser(), "--calib-data"

    for attr in ("calibration_data", "calib_data", "calibration_root", "data_root"):
        raw = getattr(cfg, attr, None)
        if raw:
            return Path(raw).expanduser(), f"cfg.{attr}"

    env_value = env_map.get("COBRA_CALIB_DATA")
    if env_value:
        return Path(env_value).expanduser(), "env.COBRA_CALIB_DATA"

    return None, None


def _discover_image_paths(source: Path) -> List[Path]:
    if source.is_file():
        return [source]
    if not source.exists():
        return []
    return sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in _CALIB_IMAGE_EXTENSIONS and path.is_file()
    )


def _expand_to_limit(paths: List[Path], limit: Optional[int]) -> List[Path]:
    if limit is None or limit <= 0 or not paths:
        return paths
    if len(paths) >= limit:
        return paths[:limit]
    repeats = math.ceil(limit / len(paths))
    expanded = list(paths) * repeats
    return expanded[:limit]


def _log_real_quant_stage(model: nn.Module, stage: str) -> None:
    observer_stats = count_observers(model)
    total_quantizers = sum(
        1 for module in model.modules() if isinstance(module, UniformAffineQuantizer)
    )
    pending_quantizers = count_uninitialized_quantizers(model)
    finalized_quantizers = max(total_quantizers - pending_quantizers, 0)
    message = (
        f"[RealQuant] {stage} observers_total={observer_stats['total']} "
        f"observing={observer_stats['observing']} initialized={observer_stats['initialized']} "
        f"finalized={finalized_quantizers}/{total_quantizers} pending={pending_quantizers}"
    )
    logger.info(message)
    print(message)


def _count_observers(model: nn.Module) -> types.SimpleNamespace:
    stats = count_observers(model)
    return types.SimpleNamespace(
        total=int(stats.get("total", 0)),
        observing=int(stats.get("observing", 0)),
        initialized=int(stats.get("initialized", 0)),
    )


def _histogram_quantile(hist_manager: Any, quantile: float) -> Optional[torch.Tensor]:
    if not (0.0 <= quantile <= 1.0):
        return None
    hist = getattr(hist_manager, "hists_mat", None)
    edges = getattr(hist_manager, "bin_edges_mat", None)
    if not isinstance(hist, torch.Tensor) or hist.numel() == 0:
        return None
    if not isinstance(edges, torch.Tensor) or edges.numel() == 0:
        return None
    hist_cpu = hist.detach().float().cpu()
    edges_cpu = edges.detach().float().cpu()
    num_bins = hist_cpu.shape[-1]
    if num_bins <= 0 or edges_cpu.shape[-1] != num_bins + 1:
        return None
    results: List[float] = []
    for row_hist, row_edges in zip(hist_cpu, edges_cpu):
        total = float(row_hist.sum().item())
        if total <= 0.0:
            results.append(float("nan"))
            continue
        target = float(quantile) * total
        cumulative = torch.cumsum(row_hist, dim=0)
        idx_tensor = torch.searchsorted(cumulative, torch.tensor(target), right=False)
        idx = int(idx_tensor.item())
        if idx >= num_bins:
            idx = num_bins - 1
        prev_cum = float(cumulative[idx - 1].item()) if idx > 0 else 0.0
        bin_count = float(row_hist[idx].item())
        left_edge = float(row_edges[idx].item())
        right_edge = float(row_edges[idx + 1].item())
        if bin_count <= 0.0:
            value = right_edge
        else:
            fraction = (target - prev_cum) / bin_count
            fraction = max(0.0, min(1.0, fraction))
            value = left_edge + fraction * (right_edge - left_edge)
        results.append(value)
    return torch.tensor(results)


def _format_optional_float(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.6g}"


def _format_percent_value(value: Optional[object]) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        try:
            return "/".join(f"{float(item):.6g}" for item in value)
        except (TypeError, ValueError):
            return "/".join(str(item) for item in value)
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _summarize_activation_quantizers(
    quantizer_map: Mapping[int, tuple[UniformAffineQuantizer, str]],
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    for qid, (quantizer, label) in quantizer_map.items():
        if ".activation" not in label:
            continue
        entry: Dict[str, object] = {"label": label, "percent": None}
        observer = getattr(quantizer, "observer", None)
        min_tensor = None
        max_tensor = None
        if observer is not None:
            min_tensor = getattr(observer, "min_val", None)
            max_tensor = getattr(observer, "max_val", None)
        min_value: Optional[float] = None
        max_value: Optional[float] = None
        if isinstance(min_tensor, torch.Tensor) and min_tensor.numel() > 0:
            min_value = float(min_tensor.min().item())
        if isinstance(max_tensor, torch.Tensor) and max_tensor.numel() > 0:
            max_value = float(max_tensor.max().item())
        scale_tensor = getattr(quantizer, "scale", None)
        if (min_value is None or max_value is None) and isinstance(scale_tensor, torch.Tensor) and scale_tensor.numel() > 0:
            scale_cpu = scale_tensor.detach().float().cpu().reshape(-1)
            qmin = float(getattr(quantizer, "qmin", -128))
            qmax = float(getattr(quantizer, "qmax", 127))
            zero_tensor = getattr(quantizer, "round_zero_point", None)
            if isinstance(zero_tensor, torch.Tensor) and zero_tensor.numel() > 0:
                zero_cpu = zero_tensor.detach().float().cpu().reshape(-1).to(scale_cpu)
                candidate_min = (qmin - zero_cpu) * scale_cpu
                candidate_max = (qmax - zero_cpu) * scale_cpu
                if min_value is None:
                    min_value = float(candidate_min.min().item())
                if max_value is None:
                    max_value = float(candidate_max.max().item())
            else:
                span = scale_cpu * max(abs(qmin), abs(qmax))
                if min_value is None:
                    min_value = float((-span).min().item())
                if max_value is None:
                    max_value = float(span.max().item())
        entry["min"] = min_value
        entry["max"] = max_value
        entry["dynamic_range"] = None
        if min_value is not None and max_value is not None:
            entry["dynamic_range"] = max_value - min_value
        percent_attr = getattr(quantizer, "percent", None)
        if percent_attr is not None:
            entry["percent"] = percent_attr
        clamped_count: Optional[int] = None
        p99_low = p99_high = None
        p999_low = p999_high = None
        hist_manager = getattr(observer, "hist_manager", None) if observer is not None else None
        if hist_manager is not None:
            hist = getattr(hist_manager, "hists_mat", None)
            edges = getattr(hist_manager, "bin_edges_mat", None)
            if isinstance(hist, torch.Tensor) and hist.numel() > 0 and isinstance(edges, torch.Tensor) and edges.numel() > 0:
                hist_cpu = hist.detach().float().cpu()
                edges_cpu = edges.detach().float().cpu()
                centers = (edges_cpu[:, 1:] + edges_cpu[:, :-1]) / 2
                if isinstance(min_tensor, torch.Tensor) and min_tensor.numel() > 0:
                    min_vals = min_tensor.detach().float().cpu().reshape(-1, 1)
                elif min_value is not None:
                    min_vals = torch.full((hist_cpu.shape[0], 1), float(min_value))
                else:
                    min_vals = None
                if isinstance(max_tensor, torch.Tensor) and max_tensor.numel() > 0:
                    max_vals = max_tensor.detach().float().cpu().reshape(-1, 1)
                elif max_value is not None:
                    max_vals = torch.full((hist_cpu.shape[0], 1), float(max_value))
                else:
                    max_vals = None
                if min_vals is not None and max_vals is not None:
                    left = torch.where(centers < min_vals, hist_cpu, torch.zeros_like(hist_cpu))
                    right = torch.where(centers > max_vals, hist_cpu, torch.zeros_like(hist_cpu))
                    clamped_count = int((left + right).sum().item())
                p99_low_tensor = _histogram_quantile(hist_manager, 0.01)
                p99_high_tensor = _histogram_quantile(hist_manager, 0.99)
                if p99_low_tensor is not None and p99_low_tensor.numel() > 0:
                    p99_low = float(p99_low_tensor.min().item())
                if p99_high_tensor is not None and p99_high_tensor.numel() > 0:
                    p99_high = float(p99_high_tensor.max().item())
                p999_low_tensor = _histogram_quantile(hist_manager, 0.001)
                p999_high_tensor = _histogram_quantile(hist_manager, 0.999)
                if p999_low_tensor is not None and p999_low_tensor.numel() > 0:
                    p999_low = float(p999_low_tensor.min().item())
                if p999_high_tensor is not None and p999_high_tensor.numel() > 0:
                    p999_high = float(p999_high_tensor.max().item())
        entry["clamped_count"] = clamped_count
        entry["p99"] = (p99_low, p99_high)
        entry["p999"] = (p999_low, p999_high)
        entry["quantizer_id"] = qid
        entries.append(entry)
    entries.sort(key=lambda item: item["label"])
    return entries


def _print_activation_summary(
    entries: Sequence[Dict[str, object]],
    stage: str,
    *,
    limit: int = 10,
) -> None:
    if not entries:
        message = f"[RealQuant] {stage} activation summary: no activation quantizers recorded."
        logger.info(message)
        print(message)
        return
    min_candidates = [entry["min"] for entry in entries if entry.get("min") is not None]
    max_candidates = [entry["max"] for entry in entries if entry.get("max") is not None]
    range_min = min(min_candidates) if min_candidates else None
    range_max = max(max_candidates) if max_candidates else None
    range_span = None
    if range_min is not None and range_max is not None:
        range_span = range_max - range_min
    percent_values = {
        _format_percent_value(entry.get("percent")) for entry in entries if entry.get("percent") is not None
    }
    percent_values.discard("n/a")
    percent_summary = ", ".join(sorted(percent_values)) if percent_values else "n/a"
    total_clamped = sum(entry.get("clamped_count", 0) or 0 for entry in entries)
    summary_line = (
        f"[RealQuant] {stage} replay stats: "
        f"clamped_total={total_clamped} "
        f"range=[{_format_optional_float(range_min)}, {_format_optional_float(range_max)}] "
        f"span={_format_optional_float(range_span)} "
        f"percent={percent_summary}"
    )
    logger.info(summary_line)
    print(summary_line)
    for entry in entries[:limit]:
        label = entry["label"]
        min_str = _format_optional_float(entry.get("min"))
        max_str = _format_optional_float(entry.get("max"))
        p99_low, p99_high = entry.get("p99", (None, None))
        p999_low, p999_high = entry.get("p999", (None, None))
        p99_str = f"({_format_optional_float(p99_low)}, {_format_optional_float(p99_high)})"
        p999_str = f"({_format_optional_float(p999_low)}, {_format_optional_float(p999_high)})"
        clamped_value = entry.get("clamped_count")
        clamped_str = str(clamped_value) if clamped_value is not None else "n/a"
        percent_str = _format_percent_value(entry.get("percent"))
        detail_line = (
            f"[RealQuant]   - {label}: min={min_str} max={max_str} "
            f"p99={p99_str} p99.9={p999_str} clamped={clamped_str} percent={percent_str}"
        )
        logger.info(detail_line)
        print(detail_line)


def _raise_on_uninitialized_quantizers(model: nn.Module, stage: str, *, limit: int = 10) -> None:
    missing = _gather_uninitialized_quantizers(model)
    if not missing:
        return
    header = (
        f"[RealQuant][Error] {len(missing)} quantizer(s) remain uninitialized after {stage}. "
        "Check calibration flow."
    )
    logger.error(header)
    print(header)
    for entry in missing[:limit]:
        detail = f"  - {entry}"
        logger.error(detail)
        print(detail)
    remaining = len(missing) - limit
    if remaining > 0:
        tail = f"  ... (+{remaining} more)"
        logger.error(tail)
        print(tail)
    raise RuntimeError(
        f"Uninitialized quantizers detected after {stage}; "
        "ensure observers collected scale/zero statistics before finalizing."
    )


_OBSERVER_UPDATE_COUNTS: Dict[int, int] = defaultdict(int)


def _quantizer_initialized(quantizer: UniformAffineQuantizer) -> bool:
    scale = getattr(quantizer, "scale", None)
    scale_ready = isinstance(scale, torch.Tensor) and scale.numel() > 0
    if not scale_ready:
        return False
    zero_not_required = getattr(quantizer, "disable_zero_point", False) or getattr(quantizer, "symmetric", False)
    if zero_not_required:
        return True
    for attr in ("zero_point", "round_zero_point", "zero"):
        value = getattr(quantizer, attr, None)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            return True
    return False


def _gather_uninitialized_quantizers(model: nn.Module) -> List[str]:
    missing: List[str] = []
    for module_name, module in _iter_named_modules(model):
        if not module_name:
            continue
        for role, attr in (("weight", "weight_quantizer"), ("activation", "act_quantizer")):
            quantizers = _iter_uniform_quantizers(getattr(module, attr, None))
            if not quantizers:
                continue
            attr_name = "weight_quantizer" if role.startswith("weight") else "act_quantizer"
            for quantizer in quantizers:
                if _quantizer_initialized(quantizer):
                    continue
                missing.append(_canonical_quant_key(module_name, attr_name))
    return missing


def _collect_quantizers_with_labels(model: nn.Module) -> Dict[int, tuple[UniformAffineQuantizer, str]]:
    quantizers: Dict[int, tuple[UniformAffineQuantizer, str]] = {}
    for module_name, module in _iter_named_modules(model):
        if not module_name:
            continue
        for role, attr in (("weight", "weight_quantizer"), ("activation", "act_quantizer")):
            entries = _iter_uniform_quantizers(getattr(module, attr, None))
            if not entries:
                continue
            attr_name = "weight_quantizer" if role.startswith("weight") else "act_quantizer"
            for quantizer in entries:
                qid = id(quantizer)
                if qid in quantizers:
                    continue
                quantizers[qid] = (quantizer, _canonical_quant_key(module_name, attr_name))
    return quantizers


def _prepare_observer_tracking(model: nn.Module) -> tuple[Dict[int, tuple[UniformAffineQuantizer, str]], List[tuple[Any, Any]]]:
    _OBSERVER_UPDATE_COUNTS.clear()
    wrapped: List[tuple[Any, Any]] = []
    seen_observers: Set[int] = set()
    quantizers = _collect_quantizers_with_labels(model)
    for qid, (quantizer, _label) in quantizers.items():
        observer = getattr(quantizer, "observer", None)
        update_fn = getattr(observer, "update", None) if observer is not None else None
        if observer is None or not callable(update_fn):
            continue
        obs_id = id(observer)
        if obs_id in seen_observers:
            continue
        seen_observers.add(obs_id)
        original_update = update_fn

        def _wrapped_update(self, *args, __orig=original_update, __qid=qid, **kwargs):
            _OBSERVER_UPDATE_COUNTS[__qid] += 1
            return __orig(*args, **kwargs)

        observer.update = types.MethodType(_wrapped_update, observer)
        wrapped.append((observer, original_update))
    return quantizers, wrapped


def _run_fake_quant_calib(
    model: nn.Module,
    calib_loader: DataLoader | None,
    warmup: int,
    calib_steps: int,
    cfg: Optional[QuantConfig] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    *,
    propagate_int: bool = False,
    use_int_kernel: bool = False,
    calibrate_weight: bool = True,
    calibrate_act: bool = True,
    real_quant: bool = False,
) -> tuple[int, Optional[Dict[int, tuple[UniformAffineQuantizer, str]]]]:
    from cobra.quantize.utils import (
        enable_observation,
        finalize_all_quantizers,
        set_quant_state,
        set_observing,
    )

    if calib_loader is None:
        raise RuntimeError("[Calib] calib_loader is None; provide --calib-data for Step 2 fake-quant init.")

    cfg_ref = cfg or getattr(model, "quant_config", None)
    if cfg_ref is None:
        raise RuntimeError("[Calib] QuantConfig context missing; cannot prepare fake-quant calibration inputs.")

    try:
        first_param = next(model.parameters())
    except StopIteration:
        first_param = None

    if device is None:
        if first_param is not None:
            device = first_param.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dtype is None:
        if first_param is not None:
            dtype = first_param.dtype
        else:
            dtype = torch.float32

    prev_training = model.training
    model.eval()

    def _quantizer_has_scale(quantizer: UniformAffineQuantizer) -> bool:
        scale = getattr(quantizer, "scale", None)
        return isinstance(scale, torch.Tensor) and scale.numel() > 0

    def _iter_weight_quantizers() -> Iterable[tuple[str, nn.Module, UniformAffineQuantizer]]:
        for module_name, module in _iter_named_modules(model):
            quantizers = _iter_uniform_quantizers(getattr(module, "weight_quantizer", None))
            if not quantizers:
                continue
            for quantizer in quantizers:
                yield module_name, module, quantizer

    def _preinitialize_weight_quantizers() -> int:
        initialized = 0
        for _, module, quantizer in _iter_weight_quantizers():
            if _quantizer_has_scale(quantizer):
                continue
            weight_tensor = getattr(module, "weight", None)
            if not isinstance(weight_tensor, torch.Tensor):
                continue
            if hasattr(quantizer, "init_from_weight"):
                try:
                    quantizer.init_from_weight(weight_tensor)
                    initialized += 1
                    continue
                except Exception:
                    pass
            if hasattr(quantizer, "calculate_qparams"):
                try:
                    quantizer.calculate_qparams()
                    initialized += 1
                except Exception:
                    pass
        return initialized

    def _fallback_initialize_weight_quantizers() -> int:
        recovered = 0
        for module_name, module, quantizer in _iter_weight_quantizers():
            if _quantizer_has_scale(quantizer):
                continue
            label = module_name or module.__class__.__name__
            logging.warning("[Calib] weight quantizer missing scale/zero after finalize: %s", label)
            weight_tensor = getattr(module, "weight", None)
            initialized = False
            if isinstance(weight_tensor, torch.Tensor) and hasattr(quantizer, "init_from_weight"):
                try:
                    quantizer.init_from_weight(weight_tensor)
                    initialized = _quantizer_has_scale(quantizer)
                except Exception as exc:
                    logging.debug("[Calib][fallback] init_from_weight failed for %s: %s", label, exc)
            if not initialized and hasattr(quantizer, "calculate_qparams"):
                try:
                    quantizer.calculate_qparams()
                    initialized = _quantizer_has_scale(quantizer)
                except Exception as exc:
                    logging.debug("[Calib][fallback] calculate_qparams failed for %s: %s", label, exc)
            if initialized:
                recovered += 1
                logging.info("[Calib][fallback] initialized weight scale for: %s", label)
        return recovered

    enable_observation(model)
    set_observing(model, observing=True)
    _activate_observers(model)
    try:
        set_quant_state(model, observer=True)
    except TypeError:
        pass

    _preinitialize_weight_quantizers()

    try:
        set_quant_state(model, weight_quant=True, act_quant=True)
    except RuntimeError as exc:
        if "scale" not in str(exc).lower():
            raise
        _preinitialize_weight_quantizers()
        set_quant_state(model, weight_quant=True, act_quant=True)

    effective_warmup = max(0, warmup)
    if effective_warmup > 0:
        _run_recalibration_forward(
            model,
            calib_loader,
            cfg_ref,
            device,
            dtype,
            max_batches=effective_warmup,
            quantizer_map=None,
            log_interval=0,
        )

    quantizer_map, wrapped_observers = _prepare_observer_tracking(model)

    has_real_quant_attr = hasattr(model, "use_real_quant")
    had_real = getattr(model, "use_real_quant", False)
    if has_real_quant_attr:
        setattr(model, "use_real_quant", False)

    processed = 0
    try:
        if calibrate_weight:
            set_quant_state(model, weight_quant=True, act_quant=False)
            print("[Replay] weight-init pass: set_quant_state(w=True,a=False)")
            try:
                _run_recalibration_forward(
                    model,
                    calib_loader,
                    cfg_ref,
                    device,
                    dtype,
                    max_batches=2,
                    quantizer_map=None,
                    log_interval=0,
                )
            finally:
                set_quant_state(model, weight_quant=False, act_quant=False)
                print("[Replay] weight-init pass: set_quant_state(w=False,a=False)")
        else:
            print("[Replay] Skipping weight-init pass (calibrate_weight=False)")

        finalize_all_quantizers(model, kind="weight")
        print("[Finalize] finalize_all_quantizers(kind='weight') done")
        freeze_weight_qparams(model)
        print("[Finalize] freeze_weight_qparams done")

        enable_observation(model)
        _activate_observers(model)
        print("[Replay] enable_observation done")

        set_quant_state(model, weight_quant=calibrate_weight, act_quant=calibrate_act)
        print(f"[Replay] set_quant_state(w={calibrate_weight},a={calibrate_act})")
        effective_steps = max(0, calib_steps)
        if effective_steps > 0:
            processed = _run_recalibration_forward(
                model,
                calib_loader,
                cfg_ref,
                device,
                dtype,
                max_batches=effective_steps,
                quantizer_map=quantizer_map,
            )
        else:
            print("[Replay] Skipping activation calibration pass (calib_steps=0)")
    finally:
        for observer, original_update in wrapped_observers:
            observer.update = original_update
        set_quant_state(model, weight_quant=False, act_quant=False)
        print("[Replay] set_quant_state(w=False,a=False)")
        set_observing(model, observing=False)
        finalize_all_quantizers(model, kind="activation")
        print("[Replay] finalize_all_quantizers(kind='activation') done")
        try:
            set_quant_state(model, observer=False)
        except TypeError:
            pass
        if has_real_quant_attr:
            setattr(model, "use_real_quant", had_real)

    if prev_training:
        model.train()
    else:
        model.eval()

    pre_pending_weight, pre_pending_activation = _count_pending(model)
    print(
        "[Finalize] pending before finalize_quant_params: "
        f"weight={pre_pending_weight}, activation={pre_pending_activation}"
    )

    finalize_fn = getattr(quant_pct, "finalize_quant_params", None)
    if callable(finalize_fn):
        finalize_fn(model)
    else:
        finalize_all_quantizers(model)

    _emit_calibration_logs(model)

    post_pending_weight, post_pending_activation = _count_pending(model)
    print(
        "[Finalize] pending after finalize_quant_params: "
        f"weight={post_pending_weight}, activation={post_pending_activation}"
    )
    if post_pending_weight or post_pending_activation:
        raise RuntimeError(
            "Finalization failed: "
            f"pending weight={post_pending_weight}, activation={post_pending_activation}"
        )

    _emit_percentile_snapshot(model)

    set_static_quant(model, static_quant=True)
    _guard_real_quant_export(model, real_quant=real_quant)
    if real_quant:
        register_scales_and_zeros(model)
        print("[Export] register_scales_and_zeros done")

        convert_fn = getattr(quant_pct, "convert_to_int", None)
        if callable(convert_fn):
            convert_fn(
                model,
                propagate_int=propagate_int,
                use_int_kernel=use_int_kernel,
            )
        else:
            convert_to_int(
                model,
                propagate_int=propagate_int,
                use_int_kernel=use_int_kernel,
            )
        print("[Export] convert_to_int done")
    else:
        print("[Export] Skipping INT export steps (real_quant disabled)")

    summary = summarize_quantizer_init(model)
    if summary.get("missing", 0) > 0:
        _fallback_initialize_weight_quantizers()
        summary = summarize_quantizer_init(model)

    observer_stats = count_observers(model)
    finalized = summary.get("total", 0) - summary.get("missing", 0)
    initialized = summary.get("initialized", 0)
    total_quantizers = summary.get("total", 0)

    logging.info(
        "[Calib] observers=%d/%d, replay_batches=%d, finalized=%d, initialized=%d/%d",
        observer_stats.get("observing", 0),
        observer_stats.get("total", 0),
        processed,
        finalized,
        initialized,
        total_quantizers,
    )

    set_quant_state(model, weight_quant=bool(real_quant), act_quant=bool(real_quant))
    print(f"[Run] set_quant_state(w={bool(real_quant)},a={bool(real_quant)})")

    return processed, quantizer_map


def _apply_percentile_stats_from_file(
    stats_path: Path | str,
    model: nn.Module,
    *,
    strict_missing_stats: bool = True,
    denylist: Optional[Set[str]] = None,
) -> int:
    resolved = Path(stats_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Percentile stats file '{resolved}' not found.")

    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Percentile stats at '{resolved}' must be a mapping, received {type(payload)!r}.")

    normalize_fn = getattr(quant_pct, "_normalize_stat_keys", None)
    if callable(normalize_fn):
        try:
            normalized_payload = normalize_fn(dict(payload))
            if isinstance(normalized_payload, Mapping):
                payload = normalized_payload
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("[PercentileStats] normalization via _normalize_stat_keys failed: %s", exc)

    consumed: Set[str] = set()
    missing_details: List[Tuple[str, List[str]]] = []
    applied = 0
    key_pool = [key for key in payload.keys() if isinstance(key, str)]
    denylist = set(denylist or set())

    for name, module in _iter_named_modules(model):
        if not name:
            continue

        for role, attr in (("weight", "weight_quantizer"), ("activation", "act_quantizer")):
            quantizers = [
                quant
                for quant in _iter_uniform_quantizers(getattr(module, attr, None))
                if str(getattr(quant, "mode", "")).lower() == "percentile"
            ]
            if not quantizers:
                continue
            attr_name = "weight_quantizer" if role.startswith("weight") else "act_quantizer"
            canonical_name = _rewrite_module_path(name)
            key = _canonical_quant_key(canonical_name, attr_name)
            if key in denylist:
                consumed.add(key)
                continue
            stats_entry = payload.get(key)
            legacy_key = None
            if stats_entry is None and canonical_name != name:
                legacy_key = _canonical_quant_key(name, attr_name)
                if legacy_key in denylist:
                    consumed.add(legacy_key)
                    continue
                stats_entry = payload.get(legacy_key)
                if stats_entry is not None:
                    key = legacy_key
            if stats_entry is None:
                nearest = get_close_matches(key, key_pool, n=3, cutoff=0.0)
                missing_details.append((key, nearest))
                continue
            if not isinstance(stats_entry, Mapping):
                raise TypeError(
                    f"Percentile stats for '{key}' must be a mapping, received {type(stats_entry)!r}."
                )
            for quantizer in quantizers:
                quantizer.apply_percentile_stats(stats_entry)
                applied += 1
            consumed.add(key)

    unused: List[str] = []
    for raw_key in payload.keys():
        if not isinstance(raw_key, str):
            unused.append(f"{raw_key!r} (invalid key type)")
            continue
        if raw_key.startswith("target::"):
            continue
        if raw_key not in consumed:
            unused.append(f"{raw_key} (unused)")

    missing_keys = [key for key, _ in missing_details]
    if missing_keys or unused:
        missing_preview = missing_keys[:5]
        suggest_preview = dict(missing_details[:5]) if missing_details else {}
        unresolved = len(missing_keys) + len(unused)
        if strict_missing_stats:
            raise RuntimeError(
                "[PercentileStats][fatal] "
                f"unresolved={unresolved} missing_preview={missing_preview} suggest={suggest_preview} "
                f"unused_preview={unused[:5]}"
            )
        summary = (
            f"[PercentileStats][warn] unresolved entries: missing={len(missing_keys)} unused={len(unused)} "
            f"missing_preview={missing_preview} suggest={suggest_preview} unused_preview={unused[:5]}"
        )
        print(summary)
        logger.warning(summary)

    logging.info("[PercentileStats] Applied %d percentile quantizer entries from %s.", applied, resolved)
    return applied


def _load_and_apply_percentile_stats(
    model: nn.Module,
    stats_path: str,
    *,
    strict_missing_stats: bool = True,
    denylist: Optional[Set[str]] = None,
) -> int:
    """
    讀入 Step 1 產生的百分位統計，支援單檔或資料夾輸入，並套用到模型上的 percentile quantizer。
    """


    resolved = Path(stats_path).expanduser()
    if resolved.is_dir():
        suffixes = {".pt", ".pth", ".bin", ".json"}
        candidates = sorted(
            path for path in resolved.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
        )
        if not candidates:
            raise FileNotFoundError(
                f"No percentile stats files found under '{resolved}'. Expected one of: {', '.join(sorted(suffixes))}."
            )
        applied_total = 0
        for candidate in candidates:
            applied_total += _apply_percentile_stats_from_file(
                candidate,
                model,
                strict_missing_stats=strict_missing_stats,
                denylist=denylist,
            )
        return applied_total

    return _apply_percentile_stats_from_file(
        resolved,
        model,
        strict_missing_stats=strict_missing_stats,
        denylist=denylist,
    )


def assert_finalized(model: nn.Module) -> Tuple[int, List[str]]:
    """Return the number of quantizers missing frozen parameters and a short label preview."""
    from cobra.quantize.utils import iter_quantizers

    pending: List[str] = []
    for handle in iter_quantizers(model):
        quantizer = handle.quantizer
        scale = getattr(quantizer, "scale", None)
        scale_ready = isinstance(scale, torch.Tensor) and scale.numel() > 0
        observer_active = getattr(quantizer, "observer", None) is not None
        reasons: List[str] = []
        if not scale_ready:
            reasons.append("scale")
        if observer_active:
            reasons.append("observer")
        if reasons:
            label = handle.label or f"{handle.kind}:{type(quantizer).__name__}"
            pending.append(f"{label} ({'|'.join(reasons)})")
    return len(pending), pending


def _guard_real_quant_export(model: nn.Module, *, real_quant: bool) -> Tuple[int, List[str]]:
    pending_count, pending_labels = assert_finalized(model)
    if pending_count:
        preview = ", ".join(pending_labels[:5])
        message = f"[RealQuant] Pending quantizers before export: {pending_count}"
        if preview:
            message += f" | {preview}"
        logger.warning(message)
        print(message)
        if real_quant:
            hint = "[RealQuant][Hint] 請先完成校正 forward 或移除 --real-quant。"
            print(hint)
            logger.error(hint)
            raise RuntimeError("Real-quant export blocked: quantizers not finalized.")
    return pending_count, pending_labels


def _emit_percentile_snapshot(model: nn.Module, output_path: Optional[str] = None) -> None:
    if not output_path:
        return
    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump({}, handle, indent=2, sort_keys=True)


def _count_pending(model: nn.Module) -> Tuple[int, int]:
    from cobra.quantize.utils import iter_quantizers

    pending_weight = 0
    pending_activation = 0
    for handle in iter_quantizers(model):
        if handle.kind == "weight":
            if not handle.is_initialized():
                pending_weight += 1
        elif handle.kind == "activation":
            if not handle.is_initialized():
                pending_activation += 1
    return pending_weight, pending_activation


def _count_and_fix_pending(model: nn.Module) -> Tuple[int, int]:
    from cobra.quantize.utils import iter_quantizers, finalize_all_quantizers

    weight_pending = 0
    activation_pending = 0

    for handle in iter_quantizers(model):
        if handle.kind == "weight" and not handle.is_initialized():
            weight_pending += 1
        elif handle.kind == "activation" and not handle.is_initialized():
            activation_pending += 1

    print(f"[Preflight] pending before export: weight={weight_pending}, activation={activation_pending}")

    if weight_pending > 0:
        finalize_all_quantizers(model, kind="weight")
        print("[Preflight] re-finalize weight")
    if activation_pending > 0:
        finalize_all_quantizers(model, kind="activation")
        print("[Preflight] re-finalize act")

    return weight_pending, activation_pending


def _force_finalize_all(model: nn.Module) -> None:
    """Ensure both weight and activation quantizers are finalized."""
    finalize_all_quantizers(model, kind="weight")
    finalize_all_quantizers(model, kind="activation")


def _build_calib_loader(
    args: argparse.Namespace,
    model,
    cfg: QuantConfig,
    device: torch.device,
    max_batches: int,
) -> DataLoader:
    vision = getattr(model, "vision_backbone", None)
    transform = getattr(vision, "image_transform", None) if vision is not None else None
    if transform is None:
        raise RuntimeError("[RealQuant] Vision backbone missing image_transform; cannot replay calibration.")

    root, origin = _resolve_calib_root(args, cfg)
    if root is None:
        raise RuntimeError(f"Calibration image root not specified. {_CALIB_SOURCE_GUIDANCE}")

    resolved_root = root.expanduser()
    if not resolved_root.exists():
        raise RuntimeError(
            f"Calibration image root '{resolved_root}' (source={origin}) does not exist. {_CALIB_SOURCE_GUIDANCE}"
        )

    paths = _discover_image_paths(resolved_root)
    if not paths:
        raise RuntimeError(
            f"No calibration images found under '{resolved_root}' (source={origin}). {_CALIB_SOURCE_GUIDANCE}"
        )

    limit = max_batches * cfg.batch_size if max_batches > 0 else None
    paths = _expand_to_limit(paths, limit)
    logging.info(
        "[RealQuant] Using calibration images from %s (source=%s, total=%d)",
        resolved_root,
        origin,
        len(paths),
    )

    dataset = _CalibrationDataset(paths, transform)
    if len(dataset) == 0:
        raise RuntimeError("[RealQuant] Calibration dataset is empty after processing input sources.")
    batch_size = max(1, min(cfg.batch_size, len(dataset)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_calibration,
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError("[RealQuant] Calibration dataloader produced no batches to replay.")
    return loader


def _run_recalibration_forward(
    model,
    loader: DataLoader,
    cfg: QuantConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
    quantizer_map: Optional[Dict[int, tuple[UniformAffineQuantizer, str]]] = None,
    log_interval: int = 8,
) -> int:
    processed = 0
    was_training = model.training
    model.eval()
    snapshot: Dict[int, int] = {}
    try:
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if step >= max_batches:
                    break
                if not isinstance(batch, dict):
                    continue
                pixel_values = batch.get("pixel_values")
                if pixel_values is None:
                    continue
                pixel_values = _move_to_device(pixel_values, device)
                pixel_values = _cast_float_payload(pixel_values, dtype)
                text_inputs = _extract_text_inputs(batch, model, cfg, device, pixel_values)
                model(
                    input_ids=text_inputs.get("input_ids"),
                    attention_mask=text_inputs.get("attention_mask"),
                    pixel_values=pixel_values,
                    use_cache=False,
                )
                processed += 1
                if quantizer_map and log_interval > 0 and processed % log_interval == 0:
                    updated = 0
                    for qid in quantizer_map.keys():
                        current = _OBSERVER_UPDATE_COUNTS.get(qid, 0)
                        if current > snapshot.get(qid, 0):
                            updated += 1
                        snapshot[qid] = current
                    logging.info(
                        "[RealQuant] Observer updates after %d batches: %d quantizer(s) collected new stats.",
                        processed,
                        updated,
                    )
    finally:
        if was_training:
            model.train()
    return processed


def _parse_bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enable percentile clipping with optional low-bit QuantLinear replacements."
    )
    parser.add_argument("--ckpt", required=True, help="Model identifier or local checkpoint directory.")
    parser.add_argument("--cfg", required=True, help="YAML configuration with percentile statistics path.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--question", required=True, help="Prompt or question for the model.")
    parser.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated models.")
    parser.add_argument(
        "--mode",
        default="apply",
        choices=["apply", "collect", "off"],
        help="Select clipping behaviour: apply thresholds, collect stats, or run unclipped.",
    )
    parser.add_argument(
        "--auto-recollect-on-missing",
        action="store_true",
        default=False,
        help="apply 階段若缺 observer stats，先自動跑一小輪 collect 生成對齊的 targets 再重試",
    )
    parser.add_argument(
        "--strict-missing-stats",
        action="store_true",
        default=False,
        help="Enable hard errors when percentile stats are missing during apply.",
    )
    parser.add_argument(
        "--calib-data",
        default=None,
        help="Root directory containing calibration images used for real-quant replay.",
    )
    parser.add_argument(
        "--stats-in",
        type=str,
        default=None,
        help="Path to percentile stats payload (takes precedence when provided).",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        default=None,
        help="Fallback percentile stats path used when --stats-in is not set.",
    )
    parser.add_argument(
        "--stats-out",
        type=str,
        default=None,
        help="Output path for percentile stats when running in collect mode.",
    )
    parser.add_argument(
        "--percentile-stats",
        type=str,
        default=None,
        help="Stage-level percentile stats (.pt) used to derive per-module overrides when needed.",
    )
    parser.add_argument(
        "--percentile-overrides",
        type=str,
        default=str(_DEFAULT_OVERRIDES_PATH),
        help="Path to a per-module percentile override map (.pt).",
    )
    parser.add_argument(
        "--export-best-percentile-map",
        type=str,
        default=str(_DEFAULT_OVERRIDES_PATH),
        help="Where to write the derived percentile override map during collect mode.",
    )
    parser.add_argument(
        "--policy",
        choices=("auto", "fixed"),
        default="auto",
        help="Percentile override policy passed to the builder.",
    )
    parser.add_argument(
        "--default-p",
        choices=("99.0", "99.9", "99.99", "99.999"),
        default="99.9",
        help="Fallback percentile used when deriving overrides.",
    )
    parser.add_argument(
        "--skip-recalib-when-applied",
        type=_parse_bool_arg,
        default=True,
        help="Skip short fake-quant replay when percentile overrides already populated quantizers.",
    )
    parser.add_argument(
        "--export-percentile-stats",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--lazy-init-via-fakequant",
        action="store_true",
        help="If no stats and no calib data, run a tiny fake-quant replay with synthetic inputs to initialize qparams.",
    )

    parser.add_argument(
        "--diagnose-json",
        default=None,
        help="Optional path to write diagnostic JSON summary.",
    )
    parser.add_argument(
        "--dump-activations",
        action="store_true",
        help="Dump pre/post activations and emit a JSON summary.",
    )
    parser.add_argument(
        "--dump-where",
        default="vision.dino,vision.siglip",
        help="Comma-separated list of stages to dump (vision.dino,vision.siglip,mm.out,all).",
    )
    parser.add_argument(
        "--out",
        default="outputs/debug_pct_lowbit",
        help="Directory used for activation dumps when --dump-activations is set.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(_DEFAULT_TARGET_PREFIXES),
        help=f"逗號分隔的前綴過濾；例如 {','.join(_DEFAULT_TARGET_PREFIXES)}。",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed inference repetitions for latency measurement.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup inference runs before latency measurement.",
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional override for HuggingFace cache directory (defaults to environment HF_HOME).",
    )
    parser.add_argument(
        "--datasets-dir",
        default=None,
        help="Optional override for datasets cache directory (defaults to environment DATASETS_DIR).",
    )
    parser.add_argument(
        "--weight-bits",
        type=int,
        default=None,
        help="Override default weight bit-width stored in the percentile config.",
    )
    parser.add_argument(
        "--act-bits",
        type=int,
        default=None,
        help="Override default activation bit-width stored in the percentile config.",
    )
    parser.add_argument(
        "--linear-weight-bits",
        type=int,
        default=None,
        help="Override weight bit-width specifically used when wrapping nn.Linear modules.",
    )
    parser.add_argument(
        "--linear-act-bits",
        type=int,
        default=None,
        help="Override activation bit-width specifically used when wrapping nn.Linear modules.",
    )
    parser.add_argument(
        "--skip-linear-replace",
        action="store_true",
        help="Skip replacing nn.Linear modules with QuantLinear.",
    )
    parser.add_argument(
        "--skip-conv-replace",
        action="store_true",
        help="Skip replacing convolution modules with their quantized variants.",
    )
    parser.add_argument(
        "--skip-matmul-replace",
        action="store_true",
        help="Skip replacing MatMul helpers with QuantMatMul.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during layer replacement.",
    )
    parser.add_argument(
        "--conv-weight-bits",
        type=int,
        default=None,
        help="Override convolution weight bit-width.",
    )
    parser.add_argument(
        "--conv-act-bits",
        type=int,
        default=None,
        help="Override convolution activation bit-width.",
    )
    parser.add_argument(
        "--matmul-act-bits",
        type=int,
        default=None,
        help="Override MatMul activation bit-width.",
    )
    parser.add_argument(
        "--matmul-substring",
        action="store_true",
        help="Enable substring matching when selecting MatMul helper targets.",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=64,
        help="Number of mini-batches for fake-quant replay when needed.",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=None,
        help="Override number of batches processed during fake-quant replay (default: --calib-batches).",
    )
    parser.add_argument(
        "--calibrate-weight",
        type=_parse_bool_arg,
        default=True,
        help="Enable weight quantization during fake-replay calibration (true/false, default: true).",
    )
    parser.add_argument(
        "--calibrate-act",
        type=_parse_bool_arg,
        default=True,
        help="Enable activation quantization during fake-replay calibration (true/false, default: true).",
    )
    parser.add_argument(
        "--skip-recalib",
        action="store_true",
        help="Skip replay calibration in Step 2 if stats are provided.",
    )
    parser.add_argument(
        "--real-quant",
        action="store_true",
        help="Convert supported layers to use cached INT weights/activations with automatic quant/dequant during forward.",
    )
    parser.add_argument(
        "--propagate-int",
        action="store_true",
        help="When combined with --real-quant, attempt to pass INT tensors between layers that support integer inputs.",
    )
    parser.add_argument(
        "--int-kernel",
        action="store_true",
        help="Use optimised integer kernels when --real-quant is active.",
    )
    parser.add_argument(
        "--dry-run-real-quant",
        action="store_true",
        help="Run calibration replay and parameter export without enabling INT execution.",
    )
    parser.add_argument(
        "--export-int",
        type=str,
        default=None,
        help="Directory used to export INT8 weights once real-quant conversion completes (apply mode only).",
    )
    return parser.parse_args()


def _resolve_bits(value: Optional[int], fallback: int) -> int:
    if value is None:
        return int(fallback)
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("Bit-width must be positive.")
    return resolved


def _run_replacement(
    kind: str,
    skip: bool,
    skip_message: str,
    run_message: str,
    action: Callable[[], Optional[Sequence[str]]],
) -> List[str]:
    if skip:
        if skip_message:
            print(f"[{kind}] {skip_message}")
        print(f"[{kind}] Replacement count: 0")
        return []
    print(f"[{kind}] {run_message}")
    result = action()
    if result is None:
        replacements: List[str] = []
    elif isinstance(result, list):
        replacements = result
    elif isinstance(result, (tuple, set)):
        replacements = list(result)
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
        replacements = list(result)
    else:
        replacements = []
    print(f"[{kind}] Replacement count: {len(replacements)}")
    return replacements


def main() -> None:
    args = parse_args()
    if isinstance(getattr(args, "default_p", None), str):
        try:
            args.default_p = float(args.default_p)
        except ValueError:
            args.default_p = 99.9
    deprecated_stats_flag = getattr(args, "export_percentile_stats", None)
    if deprecated_stats_flag:
        warning = (
            "[Deprecated] --export-percentile-stats is ignored; use --export-best-percentile-map instead."
        )
        print(warning)
        logger.warning(warning)
    export_request = getattr(args, "export_int", None)
    if args.mode.lower() == "collect" and not getattr(args, "stats_out", None):
        raise RuntimeError("--stats-out is required when running in collect mode.")
    if export_request and args.mode != "apply":
        message = "[export] --export-int is only supported when --mode apply is active; ignoring request."
        print(message)
        logger.warning(message)
        args.export_int = None
    elif export_request and not args.real_quant:
        message = "[export] --export-int requires --real-quant so INT weights exist; ignoring request."
        print(message)
        logger.warning(message)
        args.export_int = None
    stats_source = getattr(args, "stats_in", None) or getattr(args, "stats_path", None)
    if args.mode.lower() == "apply" and not stats_source:
        warning = "[Warning] Percentile stats not provided (--stats-in/--stats-path missing); proceeding without them."
        print(warning)
        logger.warning(warning)

    lat_repeat = max(1, int(getattr(args, "repeat", 1)))
    lat_warmup = max(0, int(getattr(args, "warmup", 0)))

    hf_home = args.hf_home or os.environ.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        print(f"[Env] HF_HOME set to {hf_home}")

    datasets_dir = args.datasets_dir or os.environ.get("DATASETS_DIR")
    if datasets_dir:
        os.environ["DATASETS_DIR"] = datasets_dir
        print(f"[Env] DATASETS_DIR set to {datasets_dir}")

    default_targets = list(_DEFAULT_TARGET_PREFIXES)
    raw_targets = getattr(args, "targets", None)
    if raw_targets is None or (isinstance(raw_targets, str) and not raw_targets.strip()):
        args.targets = default_targets.copy()
    elif isinstance(raw_targets, (list, tuple, set)) and not raw_targets:
        args.targets = default_targets.copy()

    start_time = init_peak_track()
    cfg = QuantConfig.from_file(args.cfg)
    stats_override = getattr(args, "stats_path", None)
    if stats_override:
        cfg.stats_path = str(Path(stats_override).expanduser())

    if args.weight_bits is not None:
        cfg.weight_bits = int(args.weight_bits)
    if args.act_bits is not None:
        cfg.act_bits = int(args.act_bits)

    linear_weight_bits = _resolve_bits(args.linear_weight_bits, cfg.weight_bits)
    linear_act_bits = _resolve_bits(args.linear_act_bits, cfg.act_bits)
    conv_weight_bits = _resolve_bits(args.conv_weight_bits, cfg.weight_bits)
    conv_act_bits = _resolve_bits(args.conv_act_bits, cfg.act_bits)
    matmul_act_bits = _resolve_bits(args.matmul_act_bits, cfg.act_bits)

    targets_for_apply = _normalize_apply_targets(args.targets)
    user_targets = _parse_targets(targets_for_apply)
    if user_targets is not None:
        cfg.targets = user_targets
    resolved_targets_for_log = list(user_targets) if user_targets is not None else default_targets
    print(f"[QuantTargets] Resolved target prefixes: {resolved_targets_for_log}")

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    torch_version = torch.__version__
    cuda_version = torch.version.cuda or "N/A"
    print(f"[System] torch={torch_version} cuda={cuda_version}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        free_mem = total_mem = None
        try:
            mem_info = torch.cuda.mem_get_info(current_device)
            if isinstance(mem_info, tuple) and len(mem_info) >= 2:
                free_mem, total_mem = mem_info[:2]
        except Exception:
            free_mem = total_mem = None
        props = torch.cuda.get_device_properties(current_device)
        total_mem = total_mem if total_mem is not None else props.total_memory
        if free_mem is not None:
            print(
                f"[System] GPU={gpu_name} free={free_mem / (1024 ** 3):.2f}GB total={total_mem / (1024 ** 3):.2f}GB"
            )
        else:
            print(f"[System] GPU={gpu_name} total_mem={total_mem / (1024 ** 3):.2f}GB")
    else:
        print("[System] GPU unavailable; running on CPU.")

    model = load_model(args.ckpt, hf_token=args.hf_token)
    model.to(device, dtype=dtype)

    def _ensure_quantizers_ready_for_export(stage: str) -> None:
        unresolved = _gather_uninitialized_quantizers(model)
        if not unresolved:
            return
        total_missing = len(unresolved)
        print(f"[RealQuant][Error] {total_missing} quantizer(s) remain uninitialized during {stage}.")
        for label in unresolved[:20]:
            quant_type = "weight" if ".weight" in label else "activation" if ".activation" in label else "unknown"
            print(f"  - {label} ({quant_type})")
        print("[RealQuant][Hint] 1) 確認 (A) observer_on 是否執行且未跳過重播 (--skip-recalib).")
        print(f"[RealQuant][Hint] 2) 檢查 --stats-in 路徑是否有效: {getattr(args, 'stats_in', 'N/A')}")
        print(f"[RealQuant][Hint] 3) 檢查 --targets 覆蓋是否符合期望: {getattr(args, 'targets', 'N/A')}")
        logging.error(
            "[RealQuant] Quantizer initialization failed at %s; %d pending entries.",
            stage,
            total_missing,
        )
        raise RuntimeError(f"[RealQuant] Quantizer initialization incomplete during {stage}.")

    def _apply_linear() -> List[str]:
        replacements = quant_pct.replace_linear_layers(
            model,
            cfg,
            weight_bits=linear_weight_bits,
            act_bits=linear_act_bits,
        )
        cfg.weight_bits = linear_weight_bits
        cfg.act_bits = linear_act_bits
        return replacements

    linear_replacements = _run_replacement(
        "LinearQuant",
        args.skip_linear_replace,
        "Skipping QuantLinear replacement.",
        f"Replacing nn.Linear -> QuantLinear (W{linear_weight_bits}A{linear_act_bits})",
        _apply_linear,
    )

    conv_replacements = _run_replacement(
        "ConvQuant",
        args.skip_conv_replace,
        "Skipping convolution replacement.",
        f"Replacing Conv layers (W{conv_weight_bits}A{conv_act_bits})",
        lambda: quant_pct.replace_conv_layers(
            model,
            cfg,
            weight_bits=conv_weight_bits,
            act_bits=conv_act_bits,
        ),
    )

    matmul_verbose = bool(getattr(args, "verbose", False)) and not args.skip_matmul_replace
    last_module: Optional[nn.Module] = None
    if matmul_verbose:
        module_count_before = sum(1 for _ in model.modules())
        print(
            f"[MatMulQuant][verbose] Before replacement: modules={module_count_before}, "
            "target=QuantMatMul"
        )
        original_replace_matmul = quant_pct.replace_matmul_layers

        def _tracked_replace_matmul(module: nn.Module, *r_args, **r_kwargs):
            nonlocal last_module
            last_module = module
            return original_replace_matmul(module, *r_args, **r_kwargs)

        quant_pct.replace_matmul_layers = _tracked_replace_matmul  # type: ignore[assignment]
    else:
        module_count_before = None
        original_replace_matmul = None

    matmul_replacements: List[str] = []
    try:
        matmul_replacements = _run_replacement(
            "MatMulQuant",
            args.skip_matmul_replace,
            "Skipping MatMul replacement.",
            f"Replacing MatMul helpers (A{matmul_act_bits})",
            lambda: quant_pct.replace_matmul_layers(
                model,
                cfg,
                act_bits=matmul_act_bits,
                substring_match=getattr(args, "matmul_substring", False),
            ),
        )
    except Exception as exc:
        if matmul_verbose:
            module_desc = "unknown"
            if last_module is not None:
                module_desc = f"{type(last_module).__name__} (id={id(last_module)})"
            print(f"[MatMulQuant][verbose] Failure at module {module_desc}: {exc}")
        raise
    finally:
        if matmul_verbose and original_replace_matmul is not None:
            quant_pct.replace_matmul_layers = original_replace_matmul  # type: ignore[assignment]

    if matmul_verbose and matmul_replacements:
        module_count_after = sum(1 for _ in model.modules())
        delta = module_count_after - module_count_before  # type: ignore[operator]
        print(
            f"[MatMulQuant][verbose] After replacement: modules={module_count_after} "
            f"(delta={delta})"
        )
        
    quant_linear_count = 0
    quant_conv_count = 0
    quant_matmul_count = 0
    for module in model.modules():
        if isinstance(module, QuantLinear):
            quant_linear_count += 1
        elif isinstance(module, QuantConvBase):
            quant_conv_count += 1
        elif isinstance(module, QuantMatMul):
            quant_matmul_count += 1

    total_qmods = quant_linear_count + quant_conv_count + quant_matmul_count
    print(
        "[QuantSummary] QuantLinear="
        f"{quant_linear_count}, QuantConv={quant_conv_count}, "
        f"QuantMatMul={quant_matmul_count}, total={total_qmods}"
    )
    if total_qmods == 0:
        raise RuntimeError("No Quant* modules present after replacement. Check targets or naming.")

    print(f"[QuantConfig] W{cfg.weight_bits}A{cfg.act_bits}")
    _run_replacement(
        "AuxQuant",
        False,
        "",
        "Wrapping auxiliary operators (Add/SwiGLU/Softmax/etc.)",
        lambda: quant_pct.replace_other_layers(model, cfg),
    )

    if args.dry_run_real_quant and args.real_quant:
        print("[RealQuant] --dry-run-real-quant requested; skipping INT kernel enablement but running export steps.")

    real_quant_aborted = False
    stats_loaded_early = False
    if args.real_quant or args.dry_run_real_quant:
        if args.calib_batches <= 0:
            raise ValueError("--calib-batches must be a positive integer when --real-quant is enabled.")

        max_calib_batches = args.calib_batches
        if cfg.num_batches is not None:
            max_calib_batches = min(max_calib_batches, cfg.num_batches)
        max_calib_batches = max(1, max_calib_batches)

        need_replay = True
        fake_replay_done = False
        fake_replay_processed = 0
        fake_replay_quant_map: Optional[Dict[int, tuple[UniformAffineQuantizer, str]]] = None
        synthetic_replay = False
        overrides_applied = 0
        overrides_source = getattr(args, "percentile_overrides", None) or str(_DEFAULT_OVERRIDES_PATH)
        overrides_path = Path(overrides_source).expanduser()
        primary_stats_source = getattr(args, "stats_in", None) or getattr(args, "stats_path", None)
        if overrides_path.exists():
            try:
                overrides_applied = _apply_percentile_overrides(model, str(overrides_path))
                diagnose_payload["overrides_applied"] = overrides_applied

                print(
                    f"[PercentileOverrides] Applied {overrides_applied} entries from `{overrides_path}`."
                )
                if overrides_applied > 0:
                    try:
                        overrides_payload = torch.load(overrides_path, map_location="cpu")
                    except Exception:
                        overrides_payload = {}
                    if isinstance(overrides_payload, Mapping):
                        _emit_override_report(model, overrides_payload, overrides_applied)
                        override_keys: Set[str] = set()
                        for raw_key in overrides_payload.keys():
                            if not isinstance(raw_key, str) or "." not in raw_key:
                                continue
                            module_name, attr = raw_key.rsplit(".", 1)
                            attr_name = attr.strip()
                            if attr_name not in {"weight_quantizer", "act_quantizer"}:
                                continue
                            rewritten = _rewrite_module_path(module_name)
                            try:
                                canonical_key = _canonical_quant_key(rewritten, attr_name)
                            except Exception:
                                continue
                            override_keys.add(canonical_key)
                            if rewritten != module_name:
                                try:
                                    legacy_key = _canonical_quant_key(module_name, attr_name)
                                except Exception:
                                    legacy_key = None
                                if legacy_key:
                                    override_keys.add(legacy_key)
                        if override_keys:
                            override_denylist.update(override_keys)
                    if getattr(args, "skip_recalib_when_applied", True):
                        need_replay = False
                        print("[PercentileOverrides] Skip short fake-quant replay (overrides applied).")
                else:
                    print("[PercentileOverrides] Override map contained 0 entries; continuing with replay.")
            except Exception as exc:
                print(f"[PercentileOverrides] Failed to apply overrides from {overrides_path}: {exc}")
                logger.warning("Failed to apply percentile overrides at %s: %s", overrides_path, exc, exc_info=True)
        else:
            print(
                f"[PercentileOverrides] Override map not found at {overrides_path}; "
                "will rely on calibration replay."
            )

        if primary_stats_source:
            _load_and_apply_percentile_stats(
                model,
                primary_stats_source,
                strict_missing_stats=args.strict_missing_stats,
                denylist=override_denylist or None,
            )
            stats_loaded_early = True
            diagnose_payload["stats_loaded"] = True

            if args.skip_recalib:
                need_replay = False
                print("[Step2] Loaded percentile stats from --stats-in; skip fake-quant replay.")
            elif need_replay:
                print("[Step2] Loaded stats; will still run short fake-quant replay unless --skip-recalib is set.")
        elif args.skip_recalib and not args.lazy_init_via_fakequant and overrides_applied == 0:
            raise RuntimeError(
                "[Step2] Missing overrides and stats. Provide --calib-data or supply --percentile-overrides."
            )
        elif args.lazy_init_via_fakequant:
            need_replay = True

        missing_before_initial = count_uninitialized_quantizers(model)

        calib_loader: Optional[DataLoader] = None
        if need_replay:
            warmup_batches = max(0, args.warmup)
            repeat_batches = max_calib_batches
            try:
                calib_loader = _build_calib_loader(args, model, cfg, device, max_calib_batches)
            except RuntimeError as exc:
                if not args.stats_in and args.lazy_init_via_fakequant:
                    print("[Step2] --lazy-init-via-fakequant active: using tiny synthetic calib to init qparams.")
                    length = min(2, max_calib_batches)
                    calib_loader = _build_tiny_synthetic_loader(model, cfg, device, length=max(1, length))
                    synthetic_replay = True
                    warmup_batches = 0
                    repeat_batches = max(1, length)
                else:
                    raise RuntimeError(
                        "[Step2] Missing both --stats-in and --calib-data. Use one of them or --lazy-init-via-fakequant."
                    ) from exc
            if calib_loader is None:
                if not args.stats_in and args.lazy_init_via_fakequant:
                    print("[Step2] --lazy-init-via-fakequant active: using tiny synthetic calib to init qparams.")
                    length = 2
                    calib_loader = _build_tiny_synthetic_loader(model, cfg, device, length=length)
                    synthetic_replay = True
                    warmup_batches = 0
                    repeat_batches = max(1, length)
                else:
                    raise RuntimeError("[Step2] Missing both --stats-in and --calib-data. Use one of them or --lazy-init-via-fakequant.")
            if synthetic_replay and repeat_batches > 2:
                repeat_batches = 2
            if args.calib_steps is None:
                calibration_steps = repeat_batches
            else:
                calibration_steps = max(0, int(args.calib_steps))
            stats_ready_for_apply = bool(args.mode == "apply" and (stats_loaded_early or getattr(cfg, "stats_path", None)))
            if stats_ready_for_apply:
                calibration_steps = max(calibration_steps, 1)
            fake_replay_processed, fake_replay_quant_map = _run_fake_quant_calib(
                model,
                calib_loader,
                warmup=warmup_batches,
                calib_steps=calibration_steps,
                cfg=cfg,
                device=device,
                dtype=dtype,
                propagate_int=args.propagate_int,
                use_int_kernel=args.int_kernel,
                calibrate_weight=bool(args.calibrate_weight),
                calibrate_act=bool(args.calibrate_act),
                real_quant=bool(args.real_quant),
            )

            diagnose_payload["replay_batches"] = fake_replay_processed
            fake_replay_done = True
        else:
            _force_finalize_all(model)
            _record_pending_diag()
            _raise_on_uninitialized_quantizers(model, "(B) overrides_only")
            _emit_percentile_snapshot(model)
            _guard_real_quant_export(model, real_quant=bool(args.real_quant))
            if args.real_quant:
                register_scales_and_zeros(model)
                print("[RealQuant] register_scales_and_zeros done (no replay path)")
            else:
                print("[RealQuant] Skipping register_scales_and_zeros (real_quant disabled)")

        def _apply_quant_state(weight_quant: bool, act_quant: bool, observer: Optional[bool]) -> None:
            for module in model.modules():
                setter = getattr(module, "set_quant_state", None)
                if not callable(setter):
                    continue
                try:
                    if observer is None:
                        setter(weight_quant, act_quant)
                    else:
                        setter(weight_quant, act_quant, observer=observer)
                except TypeError:
                    setter(weight_quant, act_quant)

        set_quant_state(model, weight_quant=False, act_quant=True)
        enable_observation(model)
        stats = _count_observers(model)
        if stats.observing == 0:
            message = "No observers observing; call set_quant_state(..., act_quant=True) earlier"
            logger.error(f"[RealQuant] {message}")
            raise RuntimeError(message)
        _log_real_quant_stage(model, "(0) post_replacement")

        print("[RealQuant] (A) observer_on")
        set_static_quant(model, static_quant=True)
        enable_observation(model)
        stats = _count_observers(model)
        if stats.observing == 0:
            message = "No observers observing; call set_quant_state(..., act_quant=True) earlier"
            logger.error(f"[RealQuant] {message}")
            raise RuntimeError(message)
        _log_real_quant_stage(model, "(A) observer_on")
        _apply_quant_state(False, False, observer=True)

        missing_before = missing_before_initial
        print(f"[RealQuant] Uninitialized quantizers before calibration: {missing_before}")
        pre_replay_stats = _count_observers(model)
        pre_message = (
            "[RealQuant] (B) pre-replay observers: "
            f"total={pre_replay_stats.total} observing={pre_replay_stats.observing} "
            f"initialized={pre_replay_stats.initialized}"
        )
        logger.info(pre_message)
        print(pre_message)
        print("[RealQuant] (B) replay_calibration")
        stage_label = "(B) skipped_replay"
        quantizer_map: Optional[Dict[int, tuple[UniformAffineQuantizer, str]]] = fake_replay_quant_map
        if fake_replay_done:
            stage_label = "(B) post_replay"
            print(f"[RealQuant] Replayed {fake_replay_processed} calibration batch(es).")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if quantizer_map is not None:
                total_quantizers = len(quantizer_map)
                initialized_count = sum(
                    1 for quantizer, _label in quantizer_map.values() if _quantizer_initialized(quantizer)
                )
            else:
                observer_after = _count_observers(model)
                total_quantizers = observer_after.total
                initialized_count = observer_after.initialized
            uninitialized_count = total_quantizers - initialized_count
            logger.info(
                "[RealQuant] Observer summary: total=%d initialized=%d uninitialized=%d",
                total_quantizers,
                initialized_count,
                uninitialized_count,
            )
            print(
                f"[RealQuant] Observer summary: total={total_quantizers} initialized={initialized_count} uninitialized={uninitialized_count}"
            )
            if initialized_count == 0:
                if total_quantizers == 0:
                    warning = "[RealQuant] No percentile quantizers observed; skipping observation checks."
                    print(warning)
                    logger.warning(warning)
                else:
                    warning = (
                        "[RealQuant] Observer not enabled or static quant not released; "
                        "falling back to loaded stats or lazy fake-quant."
                    )
                    print(warning)
                    logger.warning(warning)
                    if not getattr(args, "stats_in", None) and not args.lazy_init_via_fakequant:
                        hint = "[RealQuant] è«‹ç¢ºèª --calib-dataã€--calib-batches æˆ–ä½¿ç”¨ --lazy-init-via-fakequant ä½œç‚ºå¾Œå‚™ã€‚"
                        print(hint)
                        logger.warning(hint)
            set_static_quant(model, static_quant=True)
            _force_finalize_all(model)
            _record_pending_diag()
            _raise_on_uninitialized_quantizers(model, stage_label)
            _guard_real_quant_export(model, real_quant=bool(args.real_quant))
            if args.real_quant:
                register_scales_and_zeros(model)
            else:
                print("[RealQuant] Skipping register_scales_and_zeros at stage B (real_quant disabled)")
        elif args.skip_recalib:
            print("[RealQuant] --skip-recalib set; skipping observation replay.")
            set_static_quant(model, static_quant=True)
            _force_finalize_all(model)
            _record_pending_diag()
            _raise_on_uninitialized_quantizers(model, stage_label)
            _guard_real_quant_export(model, real_quant=bool(args.real_quant))
            if args.real_quant:
                register_scales_and_zeros(model)
            else:
                print("[RealQuant] Skipping register_scales_and_zeros (skip-recalib path)")
        else:
            loader = _build_calib_loader(args, model, cfg, device, max_calib_batches)
            _activate_observers(model)
            quantizer_map, wrapped_observers = _prepare_observer_tracking(model)
            try:
                processed = _run_recalibration_forward(
                    model,
                    loader,
                    cfg,
                    device,
                    dtype,
                    max_batches=max_calib_batches,
                    quantizer_map=quantizer_map,
                )
            finally:
                for observer, original_update in wrapped_observers:
                    observer.update = original_update
            stage_label = "(B) post_replay"
            print(f"[RealQuant] Replayed {processed} calibration batch(es).")
            diagnose_payload["replay_batches"] = processed
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_quantizers = len(quantizer_map)
            initialized_count = sum(
                1 for quantizer, _label in quantizer_map.values() if _quantizer_initialized(quantizer)
            )
            uninitialized_count = total_quantizers - initialized_count
            logger.info(
                "[RealQuant] Observer summary: total=%d initialized=%d uninitialized=%d",
                total_quantizers,
                initialized_count,
                uninitialized_count,
            )
            print(
                f"[RealQuant] Observer summary: total={total_quantizers} initialized={initialized_count} uninitialized={uninitialized_count}"
            )
            if initialized_count == 0:
                if total_quantizers == 0:
                    warning = "[RealQuant] No percentile quantizers observed; skipping observation checks."
                    print(warning)
                    logger.warning(warning)
                else:
                    warning = (
                        "[RealQuant] Observer not enabled or static quant not released; "
                        "falling back to loaded stats or lazy fake-quant."
                    )
                    print(warning)
                    logger.warning(warning)
                    if not getattr(args, "stats_in", None) and not args.lazy_init_via_fakequant:
                        hint = "[RealQuant] 請確認 --calib-data、--calib-batches 或使用 --lazy-init-via-fakequant 作為後備。"
                        print(hint)
                        logger.warning(hint)
            set_static_quant(model, static_quant=True)
            _force_finalize_all(model)
            _record_pending_diag()
            _raise_on_uninitialized_quantizers(model, stage_label)
            _guard_real_quant_export(model, real_quant=bool(args.real_quant))
            register_scales_and_zeros(model)
        if quantizer_map:
            activation_entries = _summarize_activation_quantizers(quantizer_map)
        else:
            activation_entries = []
        _print_activation_summary(activation_entries, stage_label)
        _log_real_quant_stage(model, stage_label)
        observer_stats = count_observers(model)
        if observer_stats["total"] == 0:
            warning = "[RealQuant] No observers detected; aborting real-quant pipeline."
            logger.warning(warning)
            print(warning)
            real_quant_aborted = True

    if args.real_quant or args.dry_run_real_quant:
        if real_quant_aborted:
            print("[RealQuant] Skipping freeze/export due to missing observers.")
        else:
            pre_register_called = False
        print("[RealQuant] (C) freeze_and_export")
        set_static_quant(model, static_quant=True)
        if getattr(args, "stats_in", None) and not stats_loaded_early:
            print(f"[RealQuant] Loading percentile stats from {args.stats_in}")
            _ = _load_and_apply_percentile_stats(
                model,
                args.stats_in,
                strict_missing_stats=args.strict_missing_stats,
                denylist=override_denylist or None,
            )
            diagnose_payload["stats_loaded"] = True
            if args.lazy_init_via_fakequant:
                remaining_after_stats = count_uninitialized_quantizers(model)
                if remaining_after_stats > 0:
                    print(
                        f"[RealQuant] Lazy fake-quant init triggered; {remaining_after_stats} quantizers still uninitialized."
                    )
                    _apply_quant_state(True, True, observer=False)
                    try:
                        lazy_batches = min(2, max(1, getattr(args, "lazy_init_batches", 1)))
                        loader = _build_calib_loader(args, model, cfg, device, lazy_batches)
                        _run_recalibration_forward(
                            model,
                            loader,
                            cfg,
                            device,
                            dtype,
                            max_batches=lazy_batches,
                            quantizer_map=None,
                            log_interval=0,
                        )
                    finally:
                        _apply_quant_state(False, False, observer=False)
                    print("[RealQuant] Lazy fake-quant init completed.")
            for module in model.modules():
                weight_quantizer = getattr(module, "weight_quantizer", None)
                if weight_quantizer is None:
                    continue
                quantizers = _iter_uniform_quantizers(weight_quantizer)
                if not quantizers:
                    continue
                weight_tensor = getattr(module, "weight", None)
                if not isinstance(weight_tensor, torch.Tensor):
                    continue
                for quant in quantizers:
                    if _quantizer_initialized(quant):
                        continue
                    quant.init_from_weight(weight_tensor)
            _force_finalize_all(model)
            _record_pending_diag()
            _raise_on_uninitialized_quantizers(model, "(C) freeze_and_export")
            if _finalize_quant_params is not finalize_all_quantizers:
                _finalize_quant_params(model)
            _force_finalize_all(model)
            _record_pending_diag()
            _ensure_quantizers_ready_for_export("pre-register_scales_and_zeros")
            _guard_real_quant_export(model, real_quant=bool(args.real_quant))
            if args.real_quant:
                register_scales_and_zeros(model)
                pre_register_called = True
            else:
                print("[RealQuant] Skipping register_scales_and_zeros (freeze/export) because real_quant is disabled")
                pre_register_called = False
            _log_real_quant_stage(model, "(C) freeze_and_export")
            _force_finalize_all(model)
            _record_pending_diag()
            _ensure_quantizers_ready_for_export("post-register_scales_and_zeros")
            missing_after = count_uninitialized_quantizers(model)
            print(f"[RealQuant] Uninitialized quantizers after calibration: {missing_after}")
            assert_all_initialized(model)
            if not pre_register_called and args.real_quant:
                _force_finalize_all(model)
                _record_pending_diag()
                _ensure_quantizers_ready_for_export("fallback-register_scales_and_zeros")
                _guard_real_quant_export(model, real_quant=bool(args.real_quant))
                register_scales_and_zeros(model)
            print("[RealQuant] (D) enable_real_int")
            if args.real_quant:
                set_quant_state(model, weight_quant=True, act_quant=False)
                set_quant_state(model, observer=True)
            _apply_quant_state(True, True, observer=False)
        if args.dry_run_real_quant:
            print("[RealQuant] Dry-run complete; INT execution not enabled.")
            if args.export_int:
                print("[export] --export-int requested but dry-run mode skipped INT conversion; nothing exported.")
        elif args.real_quant:
            print(
                "[RealQuant] Switching to INT execution "
                f"(propagate_int={args.propagate_int}, int_kernel={args.int_kernel})."
            )
            if args.int_kernel:
                logger.info("[RealQuant] Integer kernel requested; enabling for supported modules.")
            pre_quant_state = {
                "use_weight_quant": getattr(model, "use_weight_quant", None),
                "use_act_quant": getattr(model, "use_act_quant", None),
            }
            print(
                "[RealQuant] Model quant flags before switch: "
                + ", ".join(f"{k}={v}" for k, v in pre_quant_state.items())
            )
            convert_to_int(
                model,
                propagate_int=args.propagate_int,
                use_int_kernel=args.int_kernel,
            )
            post_quant_state = {
                "use_weight_quant": getattr(model, "use_weight_quant", None),
                "use_act_quant": getattr(model, "use_act_quant", None),
            }
            print(
                "[RealQuant] INT mode enabled successfully. "
                + ", ".join(f"{k}={v}" for k, v in post_quant_state.items())
            )
            if args.mode == "apply" and args.export_int:
                export_path = _export_int_weights(model, Path(args.export_int).expanduser())
                if export_path is not None:
                    message = f"[export] Exported INT8 weights to {export_path}"
                    logger.info(message)
                    print(message)
            _log_real_quant_stage(model, "(D) enable_real_int")
    elif args.int_kernel:
        print("[IntKernel] --int-kernel requested without --real-quant; ignoring.")
    lat_meter = LatencyMeter(repeat=lat_repeat)

    dump_targets: Set[str] = _parse_dump_targets(args.dump_where) if args.dump_activations else set()
    dumper = ActivationDumper(dump_targets) if args.dump_activations else None
    dump_handles: List[object] = []


    diagnose_payload = {"overrides_applied": 0, "stats_loaded": False, "pending_after_finalize": {"weight": 0, "activation": 0}, "int_mode": bool(args.real_quant), "replay_batches": 0}

    def _record_pending_diag() -> None:
        pending_weight, pending_activation = _count_pending(model)
        diagnose_payload["pending_after_finalize"] = {"weight": pending_weight, "activation": pending_activation}

    override_denylist: Set[str] = set()

    if args.mode == "off":
        if args.diagnose_json:
            _record_pending_diag()
            diagnose_path = Path(args.diagnose_json).expanduser()
            diagnose_path.parent.mkdir(parents=True, exist_ok=True)
            with diagnose_path.open("w", encoding="utf-8") as diag_file:
                json.dump(diagnose_payload, diag_file, indent=2)
            print(f"[Diagnose] wrote diagnostics to {diagnose_path}")

        quant_pct.disable(model)
        if dumper is not None:
            targets = dump_targets or _DEFAULT_DUMP_POINTS
            dump_handles = _register_passthrough_hooks(model, dumper, targets)
    else:
        if user_targets:
            enable_targets: Tuple[str, ...] = tuple(normalize_targets(user_targets))
        else:
            enable_targets = _DEFAULT_CANONICAL_TARGETS

        def _enable_apply(strict_missing: bool) -> None:
            canonical_targets = tuple(normalize_targets(enable_targets))
            hook_targets = expand_targets_for_hooks(canonical_targets)
            if not hook_targets:
                hook_targets = list(canonical_targets)

            stats_path = getattr(cfg, "stats_path", None)
            if stats_path is None:
                raise RuntimeError("[QuantPct][fatal] QuantConfig.stats_path is not configured; cannot apply clipping.")
            stats_path = Path(stats_path)
            exists = stats_path.exists()
            is_file = stats_path.is_file() if exists else False
            print(
                "[QuantPct][diagnose] "
                f"stats_path={stats_path} exists={exists} is_file={is_file} targets={canonical_targets}"
            )
            if not exists or not is_file:
                raise FileNotFoundError(
                    f"[QuantPct][fatal] Percentile stats file '{stats_path}' not found. "
                    "Run with --mode collect before applying percentiles."
                )

            try:
                stats_payload = load_stats(stats_path)
            except Exception as exc:
                raise RuntimeError(
                    f"[QuantPct][fatal] Failed to load percentile stats from '{stats_path}'."
                ) from exc

            if isinstance(stats_payload, Mapping):
                _log_stats_summary(stats_payload, stage="apply:keys_before_normalize")
                normalize_fn = getattr(quant_pct, "_normalize_stat_keys", None)
                if callable(normalize_fn):
                    stats_payload = normalize_fn(stats_payload)
                    _log_stats_summary(stats_payload, stage="apply:keys_after_normalize")
                else:
                    print("[QuantPct][apply] _normalize_stat_keys unavailable; skipping normalization summary.")
            else:
                print(
                    f"[QuantPct][apply] Stats payload type mismatch (expected Mapping, got {type(stats_payload).__name__})."
                )

            observers: Optional[Mapping[str, object]] = None
            if isinstance(stats_payload, Mapping):
                raw_observers = stats_payload.get("observers")
                if isinstance(raw_observers, Mapping):
                    observers = raw_observers
            if not observers:
                top_keys: Sequence[str] = ()
                if isinstance(stats_payload, Mapping):
                    top_keys = list(stats_payload.keys())[:8]
                raise RuntimeError(
                    "[QuantPct][fatal] Percentile stats are missing observer entries. "
                    f"path={stats_path} keys_preview={top_keys}. "
                    "Re-run with --mode collect to regenerate statistics."
                )

            observer_keys = [str(key) for key in observers.keys()]
            canonical_observers = set(normalize_targets(observer_keys))
            expected_canonical = set(normalize_targets(hook_targets))
            if canonical_observers and expected_canonical:
                missing_targets = sorted(expected_canonical.difference(canonical_observers))
                if missing_targets:
                    raise RuntimeError(
                        "[QuantPct][fatal] Percentile stats do not cover required targets. "
                        f"path={stats_path} missing={missing_targets}. "
                        "Re-run collect to align observer coverage."
                    )

            normalized_targets = stats_payload.get("normalized_targets") if isinstance(stats_payload, Mapping) else None
            if not normalized_targets:
                raise RuntimeError(
                    "[QuantPct][fatal] Percentile stats missing normalized_targets metadata. "
                    "Re-run collect with an updated pipeline to embed normalized target names."
                )
            normalized_targets = normalize_targets(normalized_targets)
            normalized_target_set = set(normalized_targets)
            missing_hook_targets = [target for target in hook_targets if target not in normalized_target_set]
            if missing_hook_targets:
                raise RuntimeError(
                    "[QuantPct][fatal] Percentile stats lack normalized target coverage for required hook targets. "
                    f"hook_targets={hook_targets} missing={missing_hook_targets}. "
                    "Re-run collect to refresh stats."
                )

            quant_pct.enable(
                model,
                cfg,
                mode="apply",
                dumper=dumper.capture if dumper is not None else None,
                targets=canonical_targets,
                strict_missing_stats=strict_missing,
                stats=stats_payload,
                apply_stats_denylist=override_denylist or None,
            )

        if args.mode == "apply":
            try:
                _enable_apply(args.strict_missing_stats)
            except KeyError as err:
                if not args.auto_recollect_on_missing:
                    raise
                print("[AutoRecollect] Missing stats detected. Running a short collect to align targets...")
                try:
                    quant_pct.enable(
                        model,
                        cfg,
                        mode="collect",
                        targets=enable_targets,
                        strict_missing_stats=False,
                        wrap_fake_quant=True,
                    )
                    try:
                        try:
                            dataloader = _build_calib_loader(args, model, cfg, device, max_batches=16)
                        except Exception as loader_exc:
                            print(
                                "[AutoRecollect] Failed to build calibration loader from data; falling back to tiny "
                                "synthetic samples."
                            )
                            logger.debug("[AutoRecollect] Loader construction error", exc_info=loader_exc)
                            dataloader = _build_tiny_synthetic_loader(model, cfg, device, length=2)
                        from cobra.switches import quant_pct as _qp

                        calibrate_kwargs = {
                            "data_iter": dataloader,
                            "max_steps": 16,
                            "act_bits": getattr(cfg, "act_bits", 8),
                            "weight_bits": getattr(cfg, "weight_bits", 8),
                            "calibration_dtype": getattr(cfg, "calibration_dtype", torch.float32),
                        }
                        _qp.calibrate_quantization(model, **calibrate_kwargs)
                    finally:
                        quant_pct.disable(model)
                except Exception:
                    raise err
                _enable_apply(True)
        else:
            if args.mode == "collect" and args.dump_activations:
                print(
                    "[QuantPct][collect] dump_activations is enabled; percentile observers remain active for stats capture."
                )
            quant_pct.enable(
                model,
                cfg,
                mode=args.mode,
                dumper=dumper.capture if dumper is not None else None,
                targets=enable_targets,
                strict_missing_stats=args.strict_missing_stats,
            )

    try:
        image = Image.open(Path(args.image)).convert("RGB")

        def run_inference():
            prompt_builder = model.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=args.question)
            prompt_text = prompt_builder.get_prompt()
            with torch.no_grad():
                return model.generate(image, prompt_text)

        response = run_inference()
        print(response)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if lat_warmup > 0:
            print(f"[Latency] Warmup runs: {lat_warmup}")
            for _ in range(lat_warmup):
                run_inference()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        print(f"[Latency] Measuring over repeat={lat_repeat}")
        start_time_perf = time.perf_counter()
        for _ in range(lat_repeat):
            run_inference()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end_time_perf = time.perf_counter()
        avg_seconds = (end_time_perf - start_time_perf) / lat_repeat
        lat_meter.records.append(avg_seconds)
        print(lat_meter.summary())
        _emit_mem_peak(args, cfg, start_time)
    finally:
        clip_values: Dict[str, object] = {}
        if dumper is not None:
            clip_values = _extract_clip_values(model)

        if args.mode == "off":
            for handle in dump_handles:
                handle.remove()
        else:
            if args.mode == "collect":
                stats_path_value = getattr(cfg, "stats_path", None)
                if stats_path_value:
                    warning = (
                        "[QuantPct][collect] Percentile stats export is deprecated; ignoring "
                        f"--stats-path={stats_path_value}."
                    )
                    print(warning)
                    logger.warning(warning)
                observer_attr = getattr(model, "_quant_pct_observers", None)
                stage_stats_payload: Optional[Dict[str, Any]] = None
                if isinstance(observer_attr, Mapping) and observer_attr:
                    collected: Dict[str, Dict[str, Any]] = {}
                    for key, observer in observer_attr.items():
                        if not hasattr(observer, "state_dict"):
                            continue
                        try:
                            state_dict = observer.state_dict()
                        except Exception as exc:  # pragma: no cover - defensive logging
                            logger.debug("[QuantPct][collect] Failed to read observer %s: %s", key, exc)
                            continue
                        canonical_key = normalize_target_name(str(key))
                        collected[canonical_key] = state_dict

                    if collected:
                        normalized_existing = (
                            list(normalize_targets(enable_targets)) if enable_targets else list(collected.keys())
                        )
                        stage_stats_payload = {
                            "observers": collected,
                            "targets": normalized_existing,
                            "config": cfg.to_dict(),
                        }

                export_map_value = getattr(args, "export_best_percentile_map", None)
                if export_map_value:
                    export_path = str(Path(export_map_value).expanduser())
                    overrides_payload = build_percentile_overrides(
                        model,
                        stage_stats_payload,
                        policy=getattr(args, "policy", "auto"),
                        default_p=getattr(args, "default_p", 99.9),
                    )
                    _save_percentile_overrides(export_path, overrides_payload)
                    print(
                        f"[PercentileOverrides] Exported {len(overrides_payload)} entries to `{export_path}`."
                    )
        quant_pct.disable(model)

        out_dir_value = getattr(args, "out", None)
        out_dir = Path(out_dir_value) if out_dir_value else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        if dumper is not None and out_dir is not None:
            dumper.save(out_dir, clip_values=clip_values, mode=args.mode)


if __name__ == "__main__":
    main()

