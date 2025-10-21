"""Calibration utilities for percentile-based clipping."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .config import QuantConfig
from .observers import PercentileObserver



def _move_to_device(payload, device: torch.device):
    if isinstance(payload, dict):
        return {k: _move_to_device(v, device) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_move_to_device(v, device) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_move_to_device(v, device) for v in payload)
    if isinstance(payload, torch.Tensor):
        return payload.to(device)
    raise TypeError(f"Unsupported payload type `{type(payload)}` for device transfer.")


def _ensure_text_batch(model, batch_size: int, cfg: QuantConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    tokenizer = model.llm_backbone.tokenizer
    encoded = tokenizer(
        [cfg.prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {
        "input_ids": encoded.input_ids.to(device),
        "attention_mask": encoded.attention_mask.to(device),
    }


def _find_first_tensor(payload: Any) -> Optional[torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            result = _find_first_tensor(value)
            if result is not None:
                return result
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            result = _find_first_tensor(value)
            if result is not None:
                return result
    return None


def _infer_batch_size(batch: Dict[str, Any], pixel_values: Any) -> int:
    tensor = _find_first_tensor(batch)
    if tensor is None:
        tensor = _find_first_tensor(pixel_values)
    if tensor is None or tensor.ndim == 0:
        raise ValueError("Unable to infer batch size from calibration batch.")
    return tensor.shape[0]


def _cast_float_payload(payload: Any, dtype: torch.dtype) -> Any:
    if isinstance(payload, dict):
        return {k: _cast_float_payload(v, dtype) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_cast_float_payload(v, dtype) for v in payload]
    if isinstance(payload, tuple):
        return tuple(_cast_float_payload(v, dtype) for v in payload)
    if isinstance(payload, torch.Tensor) and torch.is_floating_point(payload):
        return payload.to(dtype=dtype)
    return payload


def _extract_text_inputs(
    batch: Dict,
    model,
    cfg: QuantConfig,
    device: torch.device,
    pixel_values: Any,
) -> Dict[str, torch.Tensor]:
    if "input_ids" in batch and batch["input_ids"] is not None:
        inputs = {"input_ids": batch["input_ids"].to(device)}
        if "attention_mask" in batch and batch["attention_mask"] is not None:
            inputs["attention_mask"] = batch["attention_mask"].to(device)
        return inputs
    batch_size = _infer_batch_size(batch, pixel_values)
    return _ensure_text_batch(model, batch_size, cfg, device)


def calibrate_model(
    model,
    dataloader: DataLoader,
    cfg: QuantConfig,
    observer_factory=PercentileObserver,
    targets: Optional[Iterable[str]] = None,
) -> Dict[str, Dict]:
    reference_param = next(model.parameters())
    device = torch.device(cfg.device) if cfg.device else reference_param.device
    model_dtype = reference_param.dtype
    model.eval()

    target_list: Sequence[str]
    if targets is not None:
        target_list = tuple(targets)
    elif cfg.targets:
        target_list = tuple(cfg.targets)
    else:
        from cobra.integration.hooks import DEFAULT_PERCENTILE_TARGETS  # local import to avoid cycles

        target_list = tuple(DEFAULT_PERCENTILE_TARGETS)

    def _make_observer(name: str) -> PercentileObserver:
        try:
            return observer_factory(cfg.p_max, cfg.mode, cfg.max_samples, target=name)
        except TypeError:
            observer = observer_factory(cfg.p_max, cfg.mode, cfg.max_samples)
            if hasattr(observer, "target"):
                observer.target = name
            return observer

    observers = {name: _make_observer(name) for name in target_list}

    from cobra.integration.hooks import attach_percentile_hooks

    handles = attach_percentile_hooks(
        model,
        observers=observers,
        apply_clipping=False,
        targets=target_list,
    )

    try:
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if cfg.num_batches is not None and step >= cfg.num_batches:
                    break
                if not isinstance(batch, dict):
                    raise TypeError("Calibration dataloader must yield dictionaries.")

                pixel_values = batch.get("pixel_values")
                if pixel_values is None:
                    raise KeyError("Batch is missing `pixel_values`.")
                pixel_values = _move_to_device(pixel_values, device)
                pixel_values = _cast_float_payload(pixel_values, model_dtype)
                text_inputs = _extract_text_inputs(batch, model, cfg, device, pixel_values)

                model(
                    input_ids=text_inputs.get("input_ids"),
                    attention_mask=text_inputs.get("attention_mask"),
                    pixel_values=pixel_values,
                    use_cache=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    stats = {
        "config": cfg.to_dict(),
        "targets": list(target_list),
        "observers": {name: observers[name].state_dict() for name in target_list},
    }
    save_stats(stats, cfg.stats_path)
    return stats


def save_stats(stats: Dict, path: Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, output_path)


def load_stats(path: Path) -> Dict:
    return torch.load(Path(path), map_location="cpu")

