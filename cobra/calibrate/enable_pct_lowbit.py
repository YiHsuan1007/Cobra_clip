"""CLI entrypoint to enable percentile clipping with low-bit linear quantization."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
from PIL import Image

from cobra import load as load_model
from cobra.quantize.config import QuantConfig
from cobra.switches import quant_pct
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
    return parser.parse_args()


def _resolve_bits(value: Optional[int], fallback: int) -> int:
    if value is None:
        return int(fallback)
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("Bit-width must be positive.")
    return resolved


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

    if args.skip_linear_replace:
        print("[LinearQuant] Skipping QuantLinear replacement.")
    else:
        print(f"[LinearQuant] Replacing nn.Linear -> QuantLinear (W{linear_weight_bits}A{linear_act_bits})")
        quant_pct.replace_linear_layers(
            model,
            cfg,
            weight_bits=linear_weight_bits,
            act_bits=linear_act_bits,
        )
        cfg.weight_bits = linear_weight_bits
        cfg.act_bits = linear_act_bits

    print(f"[QuantConfig] W{cfg.weight_bits}A{cfg.act_bits}")
    quant_pct.replace_other_layers(model, cfg)
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

