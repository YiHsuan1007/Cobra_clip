"""Shared helpers for normalising percentile target prefixes and aliases."""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

# Canonical prefixes exposed to CLI users.
CANONICAL_PREFIXES: Tuple[str, ...] = ("vision_backbone", "llm_backbone", "projector")

# Mapping of fully-qualified target keys to acceptable aliases found in legacy stats.
_TARGET_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "vision_backbone.dino": ("vision_backbone.dino", "vision.dino", "dino"),
    "vision_backbone.siglip": ("vision_backbone.siglip", "vision.siglip", "siglip"),
    "vision_backbone": ("vision_backbone", "vision", "vision.dino", "vision.siglip"),
    "projector.out": ("projector.out", "projector", "mm.out", "fused", "mm"),
    "projector": ("projector", "mm.out", "fused", "mm"),
    "llm_backbone": ("llm_backbone", "llm", "mamba"),
}

# Prefix level aliases used when rewriting dotted keys.
_PREFIX_ALIAS: Dict[str, str] = {
    "vision": "vision_backbone",
    "vision_backbone": "vision_backbone",
    "mm": "projector",
    "projector": "projector",
    "llm": "llm_backbone",
    "llm_backbone": "llm_backbone",
    "fused": "projector",
}

# Direct key remapping table for common legacy entries.
_TARGET_NAME_REWRITE: Dict[str, str] = {
    "vision.dino": "vision_backbone.dino",
    "vision.siglip": "vision_backbone.siglip",
    "mm.out": "projector.out",
    "dino": "vision_backbone.dino",
    "siglip": "vision_backbone.siglip",
    "fused": "projector.out",
    "mm": "projector",
}

# Mapping from canonical keys to hook expansion targets.
_HOOK_EXPANSION: Dict[str, Tuple[str, ...]] = {
    "vision_backbone": ("vision.dino", "vision.siglip"),
    "vision_backbone.dino": ("vision.dino",),
    "vision_backbone.siglip": ("vision.siglip",),
    "projector": ("mm.out",),
    "projector.out": ("mm.out",),
    "llm_backbone": tuple(),
}


def normalize_target_name(name: str) -> str:
    """Return canonicalised percentile target name with preferred prefix."""
    raw = (name or "").strip()
    if not raw:
        return raw
    if raw in _TARGET_NAME_REWRITE:
        return _TARGET_NAME_REWRITE[raw]

    if "::" in raw:
        prefix, _, remainder = raw.partition("::")
        return f"{prefix}::{normalize_target_name(remainder)}"

    head, dot, tail = raw.partition(".")
    canonical_prefix = _PREFIX_ALIAS.get(head, head)
    if not dot:
        return canonical_prefix
    return f"{canonical_prefix}.{tail}"


def normalize_targets(targets: Iterable[str]) -> List[str]:
    """Normalise multiple targets while preserving order."""
    seen: Dict[str, None] = {}
    normalized: List[str] = []
    for target in targets:
        key = normalize_target_name(target)
        if key not in seen:
            seen[key] = None
            normalized.append(key)
    return normalized


def normalize_observer_map(observers: Mapping[str, object]) -> Dict[str, object]:
    """Re-key observer payloads using canonical names."""
    normalized: Dict[str, object] = {}
    for key, value in observers.items():
        normalized[normalize_target_name(key)] = value
    return normalized


def candidate_observer_keys(name: str) -> Tuple[str, ...]:
    """Return possible legacy keys that map to the canonical name."""
    canonical = normalize_target_name(name)
    aliases = list(_TARGET_KEY_ALIASES.get(canonical, (canonical,)))
    # Always include the canonical name first for deterministic lookup.
    if canonical not in aliases:
        aliases.insert(0, canonical)
    # Deduplicate while preserving order.
    return tuple(dict.fromkeys(aliases))


def expand_target_for_hooks(name: str) -> Tuple[str, ...]:
    """Return hook target names corresponding to the canonical target."""
    canonical = normalize_target_name(name)
    return _HOOK_EXPANSION.get(canonical, (canonical,))


def expand_targets_for_hooks(names: Sequence[str]) -> List[str]:
    """Expand a sequence of targets to hook names, preserving order and uniqueness."""
    expanded: List[str] = []
    seen: Dict[str, None] = {}
    for name in names:
        for candidate in expand_target_for_hooks(name):
            if candidate not in seen:
                seen[candidate] = None
                expanded.append(candidate)
    return expanded


def normalize_stats_payload(stats: Dict[str, object]) -> Dict[str, object]:
    """Return a stats payload with canonical target naming applied."""
    if not isinstance(stats, dict):
        return stats
    observers = stats.get("observers", {})
    if isinstance(observers, Mapping):
        stats["observers"] = normalize_observer_map(observers)
    targets = stats.get("targets")
    if isinstance(targets, Iterable) and not isinstance(targets, (str, bytes)):
        stats["targets"] = normalize_targets(targets)
    return stats

