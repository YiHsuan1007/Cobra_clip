"""CLI entrypoint to run percentile calibration."""
from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cobra import load as load_model

from cobra.quantize.calibrate import _cast_float_payload, _extract_text_inputs, _move_to_device
from cobra.quantize.config import QuantConfig
from cobra.quantize.quantizer import UniformAffineQuantizer
from cobra.quantize.percentile_aliases import normalize_target_name, expand_target_for_hooks, normalize_targets
from cobra.switches import quant_pct
from cobra.utils.mem_peak import format_block, gather_peaks, init_peak_track
from cobra.utils.latency_meter import LatencyMeter
from cobra.integration.hooks import DEFAULT_PERCENTILE_TARGET_MAP

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if hasattr(quant_pct, "rewrite_percentile_module_path"):
    _rewrite_module_path = quant_pct.rewrite_percentile_module_path
else:
    def _rewrite_module_path(path: str) -> str:
        return path


def _resolve_calibration_targets(raw_targets: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if raw_targets is None:
        return None
    expanded: List[str] = []
    seen: Dict[str, None] = {}
    for target in raw_targets:
        canonical = normalize_target_name(target)
        hook_names = expand_target_for_hooks(canonical)
        if hook_names:
            for hook_name in hook_names:
                if hook_name not in seen:
                    seen[hook_name] = None
                    expanded.append(hook_name)
        else:
            # No hook mapping available; skip to avoid downstream warnings.
            continue
    return tuple(expanded)


def _discover_images(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No images found under `{root}`.")
    return sorted(files)


class CalibrationDataset(Dataset):
    def __init__(self, root: Path, transform, limit: int | None = None) -> None:
        self.paths = _discover_images(root)
        if limit is not None:
            self.paths = self.paths[:limit]
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        image = Image.open(self.paths[idx]).convert("RGB")
        pixel_values = self.transform(image)
        return {"pixel_values": pixel_values}


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = [item["pixel_values"] for item in batch]
    first = pixel_values[0]
    if isinstance(first, dict):
        return {
            "pixel_values": {
                key: torch.stack([pv[key] for pv in pixel_values], dim=0) for key in first
            }
        }
    return {"pixel_values": torch.stack(pixel_values, dim=0)}


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


def _tensor_to_python(value: torch.Tensor) -> Any:
    tensor = value.detach().cpu()
    if tensor.numel() == 1:
        return float(tensor.item())
    return tensor.reshape(-1).tolist()


def _format_quantizer_key(module_name: str, role: str, index: int, total: int) -> str:
    role_token = "weight" if role.startswith("weight") else "act"
    if total <= 0:
        raise ValueError("`total` must be positive when formatting quantizer keys.")
    if index < 0 or index >= total:
        raise IndexError(f"Quantizer index {index} out of range for total={total}.")
    return f"{module_name}.{role_token}_quantizer.{index}"


def _safe_named_modules(model: nn.Module):
    """Yield (name, module) even if ``named_modules`` returns extra metadata."""
    for entry in model.named_modules():
        if isinstance(entry, tuple) or isinstance(entry, list):
            if len(entry) >= 2:
                yield entry[0], entry[1]
                continue
        try:
            name, module = entry  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ValueError(f"Unsupported named_modules() entry: {entry!r}") from exc
        yield name, module


def _collect_percentile_quantizer_stats(
    model: nn.Module,
    *,
    strict_missing_observer: bool = False,
) -> Tuple[OrderedDict[str, Dict[str, Any]], int]:
    stats: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    total_entries = 0
    for name, module in _safe_named_modules(model):
        if not name:
            continue
        if not any(getattr(module, attr, None) is not None for attr in ("weight_quantizer", "act_quantizer")):
            continue
        rewritten = _rewrite_module_path(name)
        for role, attr in (("weight", "weight_quantizer"), ("act", "act_quantizer")):
            quantizers = _iter_uniform_quantizers(getattr(module, attr, None))
            if not quantizers:
                continue
            total = len(quantizers)
            for idx, quant in enumerate(quantizers):
                if str(getattr(quant, "mode", "")).lower() != "percentile":
                    continue
                key = _format_quantizer_key(rewritten, role, idx, total)
                pending = bool(getattr(quant, "_pending_percentile", False))
                if pending:
                    message = f"[PercentileStats][warn] Pending observer for {key}; skipping entry."
                    if strict_missing_observer:
                        raise RuntimeError(message.replace("[PercentileStats][warn] ", ""))
                    print(message)
                    continue
                exporter = getattr(quant, "export_percentile_stats", None)
                if not callable(exporter):
                    continue
                payload = exporter()
                if not payload:
                    message = f"[PercentileStats][warn] Missing export payload for {key}; skipping entry."
                    if strict_missing_observer:
                        raise RuntimeError(message.replace("[PercentileStats][warn] ", ""))
                    print(message)
                    continue
                stats[key] = dict(payload)
                total_entries += 1
    return stats, total_entries

def _count_quantizers(model: nn.Module) -> Tuple[int, int, int]:
    weight_count = 0
    act_count = 0
    total = 0
    for _, module in _safe_named_modules(model):
        if getattr(module, "weight_quantizer", None) is not None:
            weight_count += 1
            total += 1
        if getattr(module, "act_quantizer", None) is not None:
            act_count += 1
            total += 1
    return weight_count, act_count, total


def _build_target_stats(stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    observers = stats.get("observers", {})
    export: Dict[str, Dict[str, Any]] = {}
    for target, state in observers.items():
        clip = state.get("clip")
        if clip is None:
            continue
        clip_tensor = torch.as_tensor(clip, dtype=torch.float32)
        canonical = normalize_target_name(target)
        entry: Dict[str, Any] = {
            "mode": "percentile",
            "percent": float(state.get("p_max", 0.0)),
            "numel": int(state.get("numel", 0)),
            "target": canonical,
        }
        for hook_target in expand_target_for_hooks(canonical):
            module_path = DEFAULT_PERCENTILE_TARGET_MAP.get(hook_target)
            if module_path is not None:
                entry["module"] = _rewrite_module_path(module_path)
                break
        percent_value = entry["percent"]
        percent_key = f"p{percent_value:.6g}" if percent_value else "clip"
        entry[percent_key] = _tensor_to_python(clip_tensor)
        entry["min"] = _tensor_to_python(-clip_tensor)
        entry["max"] = _tensor_to_python(clip_tensor)
        export[f"target::{canonical}"] = entry
    return export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run percentile calibration for Cobra")
    parser.add_argument("--ckpt", required=True, help="Model identifier or local checkpoint directory.")
    parser.add_argument("--data", required=True, help="Directory containing calibration images.")
    parser.add_argument("--cfg", required=True, help="YAML file describing percentile configuration.")
    parser.add_argument("--hf-token", default=None, help="Optional HuggingFace token for gated models.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated list of percentile targets to calibrate (e.g. vision.dino,vision.siglip,mm.out).",
    )
    parser.add_argument(
        "--weight_bits",
        type=int,
        default=None,
        help="Override weight bit-width used by quantization layers.",
    )
    parser.add_argument(
        "--act_bits",
        type=int,
        default=None,
        help="Override activation bit-width used by quantization layers.",
    )
    parser.add_argument(
        "--stats-out",
        default=None,
        help="Optional path to export percentile statistics for downstream application.",
    )
    parser.add_argument(
        "--wrap-fake-quant",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Wrap model with percentile-ready fake quant layers before calibration (default: enabled).",
    )
    parser.add_argument(
        "--force-percentile",
        type=float,
        choices=(99.9, 99.99, 99.999),
        default=None,
        help="Override cfg.p_max with a fixed percentile (supports 99.9, 99.99, 99.999).",
    )
    parser.add_argument(
        "--export-quantizer-sample",
        type=int,
        default=0,
        help="Print the first N per-quantizer export entries (key + clip_max) before saving stats.",
    )
    parser.add_argument(
        "--strict-missing-observer",
        action="store_true",
        help="Raise an error if any percentile quantizer lacks observer data during export.",
    )
    return parser.parse_args()


def _parse_targets(raw: Optional[str]) -> Optional[Sequence[str]]:
    if raw is None:
        return None
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return tuple(parts) if parts else None


def _emit_mem_peak(args, cfg, start_time: float) -> None:
    try:
        elapsed = time.perf_counter() - start_time
    except Exception:
        elapsed = 0.0

    percentile_repr = "mixed"
    used_fallback = True
    try:
        target_entries = getattr(cfg, "targets", None)
        percentiles: List[object] = []
        if isinstance(target_entries, (list, tuple)):
            for entry in target_entries:
                if isinstance(entry, dict):
                    if entry.get("mode") == "percentile":
                        percentiles.append(entry.get("percentile"))
        if percentiles:
            used_fallback = False
            first = percentiles[0]
            if all(p == first for p in percentiles):
                percentile_repr = str(first)
            else:
                percentile_repr = "mixed"
        if used_fallback:
            fallback = getattr(cfg, "p_max", None)
            if fallback is not None:
                percentile_repr = str(fallback)
    except Exception:
        fallback = getattr(cfg, "p_max", None)
        percentile_repr = str(fallback) if fallback is not None else "N/A"

    weight_bits = str(getattr(cfg, "weight_bits", "?"))
    act_bits = str(getattr(cfg, "act_bits", "?"))

    rotation_conf = getattr(cfg, "rotation", None)
    if isinstance(rotation_conf, dict):
        hadamard_flag = bool(rotation_conf.get("enable_hadamard", False))
        klt_flag = bool(rotation_conf.get("enable_klt", False))
    else:
        hadamard_flag = bool(getattr(rotation_conf, "enable_hadamard", False))
        klt_flag = bool(getattr(rotation_conf, "enable_klt", False))

    quant_meta = {
        "mode": getattr(args, "mode", getattr(cfg, "mode", "N/A")),
        "weight_bits": weight_bits,
        "act_bits": act_bits,
        "hadamard": hadamard_flag,
        "klt": klt_flag,
        "percentile": percentile_repr,
    }

    try:
        summary = gather_peaks()
        is_dist = False
        rank = 0
        try:
            is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
            if is_dist:
                rank = torch.distributed.get_rank()
        except Exception:
            is_dist = False
            rank = 0
        if (not is_dist) or rank == 0:
            print("")
            print(format_block(summary, quant_meta, elapsed))
    except Exception as _e:
        try:
            should_report = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                should_report = torch.distributed.get_rank() == 0
        except Exception:
            should_report = True
        if should_report:
            print(f"[mem-peak] warning: {type(_e).__name__}: {_e}")


def main() -> None:
    args = parse_args()
    start_time = init_peak_track()
    cfg = QuantConfig.from_file(args.cfg)

    if args.weight_bits is not None:
        cfg.weight_bits = int(args.weight_bits)
    if args.act_bits is not None:
        cfg.act_bits = int(args.act_bits)
    if args.force_percentile is not None:
        cfg.p_max = float(args.force_percentile)
        print(f"[QuantConfig] forcing percentile to {cfg.p_max}")
    print(f"[QuantConfig] W{cfg.weight_bits}A{cfg.act_bits}")

    user_targets = _parse_targets(args.targets)
    if user_targets is not None:
        cfg.targets = tuple(user_targets)
    calibration_targets = _resolve_calibration_targets(user_targets) if user_targets is not None else None

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype: torch.dtype
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = load_model(args.ckpt, hf_token=args.hf_token)
    model.to(device, dtype=dtype)
    if args.wrap_fake_quant:
        quant_pct.wrap_model_for_percentile(model, cfg.weight_bits, cfg.act_bits, cfg.p_max)
    else:
        print("[Calib] wrap_fake_quant disabled; assuming model already wrapped.")
    weight_q_before, act_q_before, total_q_before = _count_quantizers(model)
    if total_q_before == 0:
        warning = "[Warning] No quantizers found. Per-quantizer export will be empty."
        print(warning)
    print(
        f"[Debug][Quantizer] before-calib: weight={weight_q_before} act={act_q_before} total={total_q_before}"
    )

    transform = model.vision_backbone.image_transform
    limit = None
    if cfg.num_batches is not None:
        limit = cfg.batch_size * cfg.num_batches
    dataset = CalibrationDataset(Path(args.data), transform, limit=limit)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate,
    )
    try:
        num_batches = len(dataloader)
    except TypeError:
        num_batches = None
    if num_batches is None:
        print("[Calib] num_batches=unknown (non-sized dataloader)")
    else:
        print(f"[Calib] num_batches={num_batches}")
        if num_batches == 0:
            raise RuntimeError("[fatal] No calibration batches. Increase --num-batches or provide data.")

    fw_counter = {"value": 0}

    def _count_forward(_module, _inputs):
        fw_counter["value"] += 1

    forward_hook = model.register_forward_pre_hook(_count_forward)
    try:
        stats = quant_pct.calibrate(model, dataloader, cfg, targets=calibration_targets)
    finally:
        forward_hook.remove()
    if fw_counter["value"] == 0:
        print("[warn] forward not executed. quantizers likely remain pending.")
    weight_q_after, act_q_after, total_q_after = _count_quantizers(model)
    print(
        f"[Debug][Quantizer] before-export: weight={weight_q_after} act={act_q_after} total={total_q_after}"
    )

    if args.stats_out:
        export_payload = OrderedDict()
        export_payload["config"] = cfg.to_dict()
        export_payload["targets"] = normalize_targets(stats.get("targets") or [])
        export_payload["observers"] = stats.get("observers", {})
        quant_payload, quant_count = _collect_percentile_quantizer_stats(
            model, strict_missing_observer=args.strict_missing_observer
        )
        sample_limit = max(0, int(args.export_quantizer_sample or 0))
        if sample_limit and quant_payload:
            print(f"[QuantizerSample] previewing first {min(sample_limit, len(quant_payload))} entries:")
            for idx, (key, payload) in enumerate(quant_payload.items()):
                if idx >= sample_limit:
                    break
                clip_max = payload.get("clip_max")
                if clip_max is None:
                    clip_max = payload.get("clip")
                print(f"  - {key}: clip_max={clip_max}")
        export_payload.update(quant_payload)
        output_path = Path(args.stats_out).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(export_payload, output_path)
        if quant_count > 0:
            print(f"[PercentileStats] Exported {quant_count} quantizer entries to `{args.stats_out}`.")
        else:
            print(f"[PercentileStats][warn] No percentile quantizer entries exported to `{args.stats_out}`.")

    observed_targets = stats.get("targets") or []
    sample_summary = ", ".join(
        f"{name}: {stats['observers'][name]['numel']}" for name in observed_targets if name in stats["observers"]
    )
    print(f"Saved percentile statistics to `{cfg.stats_path}` ({sample_summary}).")

    should_report_latency = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            should_report_latency = torch.distributed.get_rank() == 0
        except Exception:
            should_report_latency = False

    if should_report_latency:
        try:
            sample_batch = next(iter(dataloader))
        except StopIteration:
            sample_batch = None
        if sample_batch is not None:
            lat_meter = LatencyMeter(repeat=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            def run_calibration_forward():
                pixel_values = sample_batch["pixel_values"]
                pixel_values_dev = _move_to_device(pixel_values, device)
                pixel_values_dev = _cast_float_payload(pixel_values_dev, dtype)
                text_inputs = _extract_text_inputs(sample_batch, model, cfg, device, pixel_values_dev)
                with torch.no_grad():
                    return model(
                        input_ids=text_inputs.get("input_ids"),
                        attention_mask=text_inputs.get("attention_mask"),
                        pixel_values=pixel_values_dev,
                        use_cache=False,
                    )

            lat_meter.measure(run_calibration_forward)
            print(lat_meter.summary())

    _emit_mem_peak(args, cfg, start_time)


if __name__ == "__main__":
    main()

