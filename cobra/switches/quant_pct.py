"""Runtime control helpers for percentile clipping."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Sequence

from torch.utils.data import DataLoader

from cobra.integration.hooks import (
    DEFAULT_PERCENTILE_TARGETS,
    attach_percentile_hooks,
    remove_handles,
)
from cobra.quantize.calibrate import calibrate_model, load_stats
from cobra.quantize.config import QuantConfig
from cobra.quantize.observers import PercentileObserver

_HANDLE_ATTR = "_quant_pct_handles"
_OBSERVER_ATTR = "_quant_pct_observers"

ActivationCallback = Callable[[str, str, object], None]

DEFAULT_TARGETS: Sequence[str] = tuple(DEFAULT_PERCENTILE_TARGETS)
LEGACY_TARGET_MAP = {
    "vision.dino": "dino",
    "vision.siglip": "siglip",
    "mm.out": "fused",
}


def calibrate(
    model,
    dataloader: DataLoader,
    cfg: QuantConfig,
    targets: Optional[Iterable[str]] = None,
) -> dict:
    """Run calibration and persist observer statistics."""
    return calibrate_model(model, dataloader, cfg, targets=targets)


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
    """Enable percentile clipping by registering forward hooks on ``model``.

    Parameters
    ----------
    model:
        Cobra model instance.
    cfg:
        Percentile configuration describing calibration statistics.
    mode:
        ``"apply"`` clamps activations using saved thresholds.
        ``"collect"`` only updates observers without clamping.
        ``"off"`` removes any existing percentile hooks.
    dumper:
        Optional callback invoked with ``(tag, phase, tensor)`` while hooks fire.
    """

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

    # Always start from a clean state before attaching new hooks.
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
