"""CLI entrypoint to enable percentile clipping and run inference."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image

from cobra import load as load_model

from cobra.quantize.config import QuantConfig
from cobra.switches import quant_pct
from cobra.utils.latency_meter import LatencyMeter
from cobra.utils.mem_peak import format_block, gather_peaks, init_peak_track

_DEFAULT_DUMP_POINTS = {"vision.dino", "vision.siglip"}
_PERCENTILES: Sequence[float] = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


class ActivationDumper:
    """Utility that stores pre/post activations and emits debug summaries."""

    def __init__(self, targets: Set[str]) -> None:
        self.targets = targets
        self.records: Dict[str, Dict[str, torch.Tensor]] = {}

    def capture(self, name: str, phase: str, tensor: torch.Tensor) -> None:
        if name not in self.targets:
            return
        self.records.setdefault(name, {})[phase] = tensor.detach().cpu()

    def save(
        self,
        out_dir: Path,
        clip_values: Optional[Dict[str, object]] = None,
        mode: Optional[str] = None,
    ) -> None:
        if not self.records:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, Dict[str, object]] = {}

        for name, phases in self.records.items():
            pre = phases.get("pre")
            if pre is None:
                continue
            post = phases.get("post")
            if post is None:
                post = pre

            torch.save(pre, out_dir / f"{name}_pre.pt")
            torch.save(post, out_dir / f"{name}_post.pt")

            entry: Dict[str, object] = {
                "pre": _tensor_stats(pre),
                "post": _tensor_stats(post),
                "clamped_count": _clamped_count(pre, post),
            }
            if clip_values and name in clip_values:
                entry["clip_value"] = clip_values[name]
            if mode is not None:
                entry["mode"] = mode
            summary[name] = entry

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _tensor_stats(tensor: torch.Tensor) -> Dict[str, object]:
    if tensor.numel() == 0:
        return {"min": 0.0, "max": 0.0, "percentiles": {}}
    flat = tensor.detach().float().reshape(-1)
    stats = {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }
    percentiles = torch.quantile(flat, torch.tensor(_PERCENTILES, dtype=torch.float32))
    stats["percentiles"] = {
        f"p{int(p * 1000) / 10:.1f}": float(val)
        for p, val in zip(_PERCENTILES, percentiles)
    }
    return stats


def _clamped_count(pre: torch.Tensor, post: torch.Tensor) -> int:
    if pre.shape != post.shape:
        post = post.view_as(pre)
    diff = ~torch.isclose(post, pre, atol=1e-6, rtol=1e-6)
    return int(diff.sum().item())


def _parse_dump_targets(raw: str) -> Set[str]:
    if not raw:
        return set()
    parts = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not parts or "all" in parts:
        return {"vision.dino", "vision.siglip", "mm.out"}
    allowed = {"vision.dino", "vision.siglip", "mm.out"}
    invalid = parts - allowed
    if invalid:
        raise ValueError("Unknown dump targets: " + ", ".join(sorted(invalid)))
    return parts


def _extract_clip_values(model) -> Dict[str, object]:
    observers = getattr(model, "_quant_pct_observers", None)
    if not observers:
        return {}
    result: Dict[str, object] = {}
    if isinstance(observers, dict):
        items = observers.items()
    else:
        names = ("vision.dino", "vision.siglip", "mm.out")
        items = zip(names, observers)
    for name, observer in items:
        clip = observer.get_clip_value()
        if clip is None:
            continue
        if isinstance(clip, torch.Tensor):
            clip_cpu = clip.detach().cpu()
            result[name] = float(clip_cpu.item()) if clip_cpu.numel() == 1 else clip_cpu.tolist()
        else:
            result[name] = float(clip)
    return result


def _register_passthrough_hooks(model, dumper: ActivationDumper, targets: Set[str]):
    handles = []

    def make(tag: str, *, is_pre: bool = False):
        @torch.no_grad()
        def hook(_module, _inputs, output):
            dumper.capture(tag, "pre", output)
            dumper.capture(tag, "post", output)
            return output

        return hook

    vision = getattr(model, "vision_backbone", None)
    if vision is not None:
        if "vision.dino" in targets and hasattr(vision, "tap_post_dino"):
            handles.append(vision.tap_post_dino.register_forward_hook(make("vision.dino")))
        if "vision.siglip" in targets and hasattr(vision, "tap_post_siglip"):
            handles.append(vision.tap_post_siglip.register_forward_hook(make("vision.siglip")))
    if "mm.out" in targets and hasattr(model, "tap_post_mm_out"):
        handles.append(model.tap_post_mm_out.register_forward_hook(make("mm.out")))
    return handles


def _parse_targets(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
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
            fallback = getattr(cfg, "p_max", getattr(cfg, "percentile", None))
            if fallback is not None:
                percentile_repr = str(fallback)
    except Exception:
        fallback = getattr(cfg, "p_max", getattr(cfg, "percentile", None))
        percentile_repr = str(fallback) if fallback is not None else "N/A"

    bits_conf = getattr(cfg, "bits", None)

    def _extract_bit(value) -> str:
        if isinstance(value, (int, float, str)):
            return str(value)
        if value is None:
            return "?"
        return str(value)

    default_weight_bits = getattr(cfg, "weight_bits", None)
    default_act_bits = getattr(cfg, "act_bits", None)
    weight_bits = _extract_bit(default_weight_bits)
    act_bits = _extract_bit(default_act_bits)
    try:
        if isinstance(bits_conf, dict):
            weight_bits = _extract_bit(
                bits_conf.get("weight", bits_conf.get("weight_bits", default_weight_bits))
            )
            act_bits = _extract_bit(
                bits_conf.get("activation", bits_conf.get("act_bits", default_act_bits))
            )
        else:
            weight_bits = _extract_bit(
                getattr(bits_conf, "weight", getattr(bits_conf, "weight_bits", default_weight_bits))
            )
            act_bits = _extract_bit(
                getattr(bits_conf, "activation", getattr(bits_conf, "act_bits", default_act_bits))
            )
    except Exception:
        weight_bits = _extract_bit(default_weight_bits)
        act_bits = _extract_bit(default_act_bits)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enable percentile clipping and run a single inference")
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
        default="outputs/debug_pct",
        help="Directory used for activation dumps when --dump-activations is set.",
    )
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated list of percentile targets to enable (e.g. vision.dino,vision.siglip,mm.out).",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = init_peak_track()
    cfg = QuantConfig.from_file(args.cfg)

    if args.weight_bits is not None:
        cfg.weight_bits = int(args.weight_bits)
    if args.act_bits is not None:
        cfg.act_bits = int(args.act_bits)
    print(f"[QuantConfig] W{cfg.weight_bits}A{cfg.act_bits}")

    user_targets = _parse_targets(args.targets)
    if user_targets is not None:
        cfg.targets = user_targets

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = load_model(args.ckpt, hf_token=args.hf_token)
    model.to(device, dtype=dtype)
    quant_pct.replace_other_layers(model, cfg)
    lat_meter = LatencyMeter(repeat=1) # Reduced repeat for quicker profiling

    dump_targets = _parse_dump_targets(args.dump_where) if args.dump_activations else set()
    dumper = ActivationDumper(dump_targets) if args.dump_activations else None
    dump_handles: List = []

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

