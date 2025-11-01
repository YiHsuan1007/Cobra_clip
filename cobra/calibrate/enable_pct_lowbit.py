"""CLI entrypoint to enable percentile clipping with low-bit linear quantization."""
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from collections import defaultdict
import types
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cobra import load as load_model
from cobra.quantize.calibrate import _cast_float_payload, _extract_text_inputs, _move_to_device
from cobra.quantize.config import QuantConfig
from cobra.quantize.quantizer import UniformAffineQuantizer
from cobra.switches import quant_pct
from cobra.quantize.utils import (
    assert_all_initialized,
    convert_to_int,
    count_uninitialized_quantizers,
    enable_observation,
    finalize_all_quantizers,
    register_scales_and_zeros,
    set_static_quant,
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


_CALIB_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_CALIB_SOURCE_GUIDANCE = (
    "Provide calibration images via CLI (--calib-data /path/to/images), "
    "QuantConfig fields (calibration_data|calib_data|calibration_root|data_root), "
    "or environment (export COBRA_CALIB_DATA=/path/to/images). Examples:\n"
    "  export COBRA_CALIB_DATA=/work/calib_images\n"
    "  python -m cobra.calibrate.enable_pct_lowbit --ckpt CKPT --cfg CONFIG "
    "--real-quant --calib-data /work/calib_images"
)

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


def _format_quantizer_key(module_name: str, role: str, index: int, total: int) -> str:
    suffix = "" if total <= 1 else f"[{index}]"
    return f"{module_name}.{role}{suffix}"


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
            total = len(quantizers)
            for idx, quantizer in enumerate(quantizers):
                if _quantizer_initialized(quantizer):
                    continue
                missing.append(_format_quantizer_key(module_name, role, idx, total))
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
            total = len(entries)
            for idx, quantizer in enumerate(entries):
                qid = id(quantizer)
                if qid in quantizers:
                    continue
                quantizers[qid] = (quantizer, _format_quantizer_key(module_name, role, idx, total))
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


def load_and_apply_percentile_stats(stats_path: Path | str, model: nn.Module) -> None:
    resolved = Path(stats_path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Percentile stats file '{resolved}' not found.")

    payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError(f"Percentile stats at '{resolved}' must be a mapping, received {type(payload)!r}.")

    consumed: Set[str] = set()
    missing: List[str] = []
    applied = 0

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
            total = len(quantizers)
            for idx, quantizer in enumerate(quantizers):
                key = _format_quantizer_key(name, role, idx, total)
                if key not in payload:
                    missing.append(key)
                    continue
                stats_entry = payload[key]
                if not isinstance(stats_entry, Mapping):
                    raise TypeError(
                        f"Percentile stats for '{key}' must be a mapping, received {type(stats_entry)!r}."
                    )
                quantizer.apply_percentile_stats(stats_entry)
                consumed.add(key)
                applied += 1

    unused: List[str] = []
    for raw_key in payload.keys():
        if not isinstance(raw_key, str):
            unused.append(f"{raw_key!r} (invalid key type)")
            continue
        if raw_key.startswith("target::"):
            continue
        if raw_key not in consumed:
            unused.append(f"{raw_key} (unused)")

    if missing or unused:
        issues = [f"{key} (missing)" for key in missing] + unused
        preview = ", ".join(issues[:20])
        raise KeyError(
            f"Percentile stats mismatch: {len(issues)} unresolved entries (first 20: {preview})."
        )

    logging.info("[PercentileStats] Applied %d percentile quantizer entries from %s.", applied, resolved)


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
        "--calib-data",
        default=None,
        help="Root directory containing calibration images used for real-quant replay.",
    )
    parser.add_argument(
        "--stats-in",
        default=None,
        help="Optional path to percentile statistics exported via calibrate_run.",
    )
    parser.add_argument(
        "--lazy-init-via-fakequant",
        action="store_true",
        help="If set, run a short fake-quant forward pass when quantizers remain uninitialized after stats load.",
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
        default=None,
        help="Comma-separated list of percentile targets to enable (e.g. vision.dino,vision.siglip,mm.out).",
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
        "--calib-batches",
        type=int,
        default=64,
        help="Calibration batches replayed prior to enabling real-quant.",
    )
    parser.add_argument(
        "--skip-recalib",
        action="store_true",
        help="Skip observation replay while still validating quantizer initialisation.",
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
    action: Callable[[], None],
) -> bool:
    if skip:
        print(f"[{kind}] {skip_message}")
        return False
    print(f"[{kind}] {run_message}")
    action()
    return True


def main() -> None:
    args = parse_args()
    start_time = init_peak_track()
    cfg = QuantConfig.from_file(args.cfg)

    if args.weight_bits is not None:
        cfg.weight_bits = int(args.weight_bits)
    if args.act_bits is not None:
        cfg.act_bits = int(args.act_bits)

    linear_weight_bits = _resolve_bits(args.linear_weight_bits, cfg.weight_bits)
    linear_act_bits = _resolve_bits(args.linear_act_bits, cfg.act_bits)
    conv_weight_bits = _resolve_bits(args.conv_weight_bits, cfg.weight_bits)
    conv_act_bits = _resolve_bits(args.conv_act_bits, cfg.act_bits)
    matmul_act_bits = _resolve_bits(args.matmul_act_bits, cfg.act_bits)

    user_targets = _parse_targets(args.targets)
    if user_targets is not None:
        cfg.targets = user_targets

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

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

    def _apply_linear() -> None:
        quant_pct.replace_linear_layers(
            model,
            cfg,
            weight_bits=linear_weight_bits,
            act_bits=linear_act_bits,
        )
        cfg.weight_bits = linear_weight_bits
        cfg.act_bits = linear_act_bits

    _run_replacement(
        "LinearQuant",
        args.skip_linear_replace,
        "Skipping QuantLinear replacement.",
        f"Replacing nn.Linear -> QuantLinear (W{linear_weight_bits}A{linear_act_bits})",
        _apply_linear,
    )

    _run_replacement(
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

    try:
        matmul_replaced = _run_replacement(
            "MatMulQuant",
            args.skip_matmul_replace,
            "Skipping MatMul replacement.",
            f"Replacing MatMul helpers (A{matmul_act_bits})",
            lambda: quant_pct.replace_matmul_layers(
                model,
                cfg,
                act_bits=matmul_act_bits,
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

    if matmul_verbose and matmul_replaced:
        module_count_after = sum(1 for _ in model.modules())
        delta = module_count_after - module_count_before  # type: ignore[operator]
        print(
            f"[MatMulQuant][verbose] After replacement: modules={module_count_after} "
            f"(delta={delta})"
        )

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

    pre_register_called = False

    if args.real_quant or args.dry_run_real_quant:
        if args.calib_batches <= 0:
            raise ValueError("--calib-batches must be a positive integer when --real-quant is enabled.")

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

        print("[RealQuant] (A) observer_on")
        set_static_quant(model, static_quant=False)
        _apply_quant_state(False, False, observer=True)

        max_calib_batches = args.calib_batches
        if cfg.num_batches is not None:
            max_calib_batches = min(max_calib_batches, cfg.num_batches)
        missing_before = count_uninitialized_quantizers(model)
        print(f"[RealQuant] Uninitialized quantizers before calibration: {missing_before}")
        print("[RealQuant] (B) replay_calibration")
        if args.skip_recalib:
            print("[RealQuant] --skip-recalib set; skipping observation replay.")
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
            print(f"[RealQuant] Replayed {processed} calibration batch(es).")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_quantizers = len(quantizer_map)
            initialized_count = sum(
                1 for quantizer, _label in quantizer_map.values() if _quantizer_initialized(quantizer)
            )
            uninitialized_count = total_quantizers - initialized_count
            logging.info(
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
                    warning = "[RealQuant] 未偵測到需要觀測的 percentile quantizer，跳過觀測統計檢查。"
                    print(warning)
                    logging.warning(warning)
                else:
                    warning = (
                        "[RealQuant] Observer 未開啟或 set_static_quant(True) 未解除，未取得觀測統計；"
                        " 後續將依賴已載入的統計或 lazy fake-quant。"
                    )
                    print(warning)
                    logging.warning(warning)
                    if not getattr(args, "stats_in", None) and not args.lazy_init_via_fakequant:
                        hint = "[RealQuant] 請確認 --calib-data、--calib-batches 或使用 --lazy-init-via-fakequant 作為後備。"
                        print(hint)
                        logging.warning(hint)
        print("[RealQuant] (C) freeze_and_export")
        set_static_quant(model, static_quant=True)
        if getattr(args, "stats_in", None):
            print(f"[RealQuant] Loading percentile stats from {args.stats_in}")
            load_and_apply_percentile_stats(args.stats_in, model)
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
        _ensure_quantizers_ready_for_export("pre-register_scales_and_zeros")
        register_scales_and_zeros(model)
        pre_register_called = True
        _ensure_quantizers_ready_for_export("post-register_scales_and_zeros")
        finalize_all_quantizers(model)
        if _finalize_quant_params is not finalize_all_quantizers:
            _finalize_quant_params(model)
        missing_after = count_uninitialized_quantizers(model)
        print(f"[RealQuant] Uninitialized quantizers after calibration: {missing_after}")
        assert_all_initialized(model)
        if not pre_register_called:
            _ensure_quantizers_ready_for_export("fallback-register_scales_and_zeros")
            register_scales_and_zeros(model)
        print("[RealQuant] (D) enable_real_int")
        _apply_quant_state(True, True, observer=False)
        if args.dry_run_real_quant:
            print("[RealQuant] Dry-run complete; INT execution not enabled.")
        elif args.real_quant:
            print(
                "[RealQuant] Switching to INT execution "
                f"(propagate_int={args.propagate_int}, int_kernel={args.int_kernel})."
            )
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
    elif args.int_kernel:
        print("[IntKernel] --int-kernel requested without --real-quant; ignoring.")
    lat_meter = LatencyMeter(repeat=1) # Reduced repeat for quicker profiling

    dump_targets: Set[str] = _parse_dump_targets(args.dump_where) if args.dump_activations else set()
    dumper = ActivationDumper(dump_targets) if args.dump_activations else None
    dump_handles: List[object] = []

    if args.mode == "off":
        quant_pct.disable(model)
        if dumper is not None:
            targets = dump_targets or _DEFAULT_DUMP_POINTS
            dump_handles = _register_passthrough_hooks(model, dumper, targets)
    else:
        quant_pct.enable(
            model,
            cfg,
            mode=args.mode,
            dumper=dumper.capture if dumper is not None else None,
            targets=user_targets,
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
        lat_meter.measure(run_inference)
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
            quant_pct.disable(model)

        if dumper is not None:
            out_dir = Path(args.out)
            dumper.save(out_dir, clip_values=clip_values, mode=args.mode)


if __name__ == "__main__":
    main()
