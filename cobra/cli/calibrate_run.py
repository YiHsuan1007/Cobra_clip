"""CLI entrypoint to run percentile calibration."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from cobra import load as load_model

from cobra.quantize.config import QuantConfig
from cobra.switches import quant_pct

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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
    return parser.parse_args()


def _parse_targets(raw: Optional[str]) -> Optional[Sequence[str]]:
    if raw is None:
        return None
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    return tuple(parts) if parts else None


def main() -> None:
    args = parse_args()
    cfg = QuantConfig.from_file(args.cfg)

    user_targets = _parse_targets(args.targets)
    if user_targets is not None:
        cfg.targets = tuple(user_targets)

    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype: torch.dtype
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = load_model(args.ckpt, hf_token=args.hf_token)
    model.to(device, dtype=dtype)

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

    stats = quant_pct.calibrate(model, dataloader, cfg, targets=user_targets)

    observed_targets = stats.get("targets") or []
    sample_summary = ", ".join(
        f"{name}: {stats['observers'][name]['numel']}" for name in observed_targets if name in stats["observers"]
    )
    print(f"Saved percentile statistics to `{cfg.stats_path}` ({sample_summary}).")


if __name__ == "__main__":
    main()
