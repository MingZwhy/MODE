"""Helpers for expert-level bit-width assignment files.

The canonical in-memory format is::

    {layer_idx: {expert_idx: bit}}

JSON files may either store this mapping directly or wrap it under
``expert_bits`` together with metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ExpertBitConfig = Dict[int, Dict[int, int]]


def parse_layer_idx(layer_key: Any) -> Optional[int]:
    """Extract a decoder layer index from an int-like key or a module path."""
    if isinstance(layer_key, int):
        return layer_key
    text = str(layer_key)
    if text.isdigit():
        return int(text)
    match = re.search(r"layers\.(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _unwrap_config(data: Mapping[str, Any]) -> Mapping[str, Any]:
    if "expert_bits" in data and isinstance(data["expert_bits"], Mapping):
        return data["expert_bits"]
    if "layers" in data and isinstance(data["layers"], Mapping):
        return data["layers"]
    return data


def normalize_expert_bit_config(data: Mapping[str, Any]) -> ExpertBitConfig:
    """Normalize JSON-like expert bit config into ``{int: {int: int}}``."""
    raw = _unwrap_config(data)
    out: ExpertBitConfig = {}
    for layer_key, layer_data in raw.items():
        layer_idx = parse_layer_idx(layer_key)
        if layer_idx is None or not isinstance(layer_data, Mapping):
            continue
        layer_bits: Dict[int, int] = {}
        for expert_key, bit in layer_data.items():
            try:
                expert_idx = int(expert_key)
                layer_bits[expert_idx] = int(bit)
            except (TypeError, ValueError):
                continue
        if layer_bits:
            out[layer_idx] = layer_bits
    return out


def load_expert_bit_config(path: str | Path) -> ExpertBitConfig:
    """Load an expert bit-width assignment JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return normalize_expert_bit_config(json.load(f))


def save_expert_bit_config(
    path: str | Path,
    expert_bits: ExpertBitConfig,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save an expert bit-width assignment with stable string keys."""
    payload = {
        "metadata": dict(metadata or {}),
        "expert_bits": {
            str(layer_idx): {
                str(expert_idx): int(bit)
                for expert_idx, bit in sorted(layer_bits.items())
            }
            for layer_idx, layer_bits in sorted(expert_bits.items())
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def get_expert_bit(
    expert_bits: Optional[Mapping[int, Mapping[int, int]]],
    layer_idx: int,
    expert_idx: int,
    default: int,
) -> int:
    """Return the configured bit-width for one expert, or ``default``."""
    if not expert_bits:
        return int(default)
    layer_bits = expert_bits.get(layer_idx)
    if not layer_bits:
        return int(default)
    return int(layer_bits.get(expert_idx, default))
