"""Hook registration helpers for percentile-based clipping."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.utils.hooks import RemovableHandle

from ..quantize import clipping
from ..quantize.observers import PercentileObserver

ActivationCallback = Callable[[str, str, torch.Tensor], None]

DEFAULT_PERCENTILE_TARGET_MAP: Dict[str, str] = {
    "vision.dino": "vision_backbone.tap_post_dino",
    "vision.siglip": "vision_backbone.tap_post_siglip",
    "mm.out": "tap_post_mm_out",
}
DEFAULT_PERCENTILE_TARGETS: Sequence[str] = tuple(DEFAULT_PERCENTILE_TARGET_MAP.keys())


def _make_activation_hook(
    observer: PercentileObserver,
    apply_clipping: bool,
    tag: str,
    dumper: Optional[ActivationCallback],
):
    @torch.no_grad()
    def hook(_module, _inputs, output):
        if dumper is not None:
            dumper(tag, "pre", output)

        if apply_clipping:
            clip = observer.get_clip_value()
            if clip is not None:
                clamped = clipping.clamp_tensor_(output, clip)
                if dumper is not None:
                    dumper(tag, "post", clamped)
                return clamped
            if dumper is not None:
                dumper(tag, "post", output)
            return output

        observer.update(output)
        if dumper is not None:
            dumper(tag, "post", output)
        return output

    return hook


def resolve_module(root, dotted: str) -> torch.nn.Module:
    module = root
    for name in dotted.split("."):
        if not hasattr(module, name):
            raise AttributeError(f"Module `{module}` does not expose attribute `{name}` while resolving `{dotted}`.")
        module = getattr(module, name)
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Resolved object `{dotted}` is not a torch.nn.Module.")
    return module


def attach_percentile_hooks(
    model,
    *,
    observers: Mapping[str, PercentileObserver],
    apply_clipping: bool,
    targets: Optional[Sequence[str]] = None,
    dumper: Optional[ActivationCallback] = None,
) -> List[RemovableHandle]:
    """Attach percentile observers to the Cobra pipeline.

    Parameters
    ----------
    model:
        Cobra VLM instance.
    observers:
        Mapping from target name to observer. Targets must exist in ``DEFAULT_PERCENTILE_TARGET_MAP``.
    apply_clipping:
        When ``True`` the observers clamp activations instead of updating statistics.
    targets:
        Optional explicit list of targets to configure. Defaults to the keys present in ``observers`` or the full
        ``DEFAULT_PERCENTILE_TARGETS`` when unspecified.
    dumper:
        Optional callback invoked with ``(tag, phase, tensor)`` to record pre/post activations.
    """

    handles: List[RemovableHandle] = []

    selected_targets: Sequence[str]
    if targets is not None:
        selected_targets = tuple(targets)
    elif observers:
        selected_targets = tuple(observers.keys())
    else:
        selected_targets = DEFAULT_PERCENTILE_TARGETS

    for name in selected_targets:
        if name not in DEFAULT_PERCENTILE_TARGET_MAP:
            raise KeyError(f"Unknown percentile target `{name}`.")
        if name not in observers:
            raise KeyError(f"Observer for target `{name}` is not provided.")
        module_path = DEFAULT_PERCENTILE_TARGET_MAP[name]
        module = resolve_module(model, module_path)
        handle = module.register_forward_hook(
            _make_activation_hook(observers[name], apply_clipping, name, dumper)
        )
        handles.append(handle)

    return handles


def remove_handles(handles: Iterable[RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()
