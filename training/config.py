from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "training"


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_training_config(path_or_robot: str | Path) -> dict[str, Any]:
    path = Path(path_or_robot)
    if not path.exists():
        path = CONFIG_ROOT / "robots" / f"{path_or_robot}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path_or_robot}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, Mapping):
        raise ValueError(f"Training config must be a mapping: {path}")
    parent = config.get("extends")
    if parent is None:
        merged = dict(config)
    else:
        merged = _merge(load_training_config(path.parent / parent), config)
    merged.pop("extends", None)
    if merged.get("schema_version") != 1:
        raise ValueError("Unsupported training config schema_version")
    if not merged.get("inputs", {}).get("low_dim"):
        raise ValueError("Training config requires inputs.low_dim")
    if merged["network"]["transformer"]["block_size"] < merged["sequence"]["length"]:
        raise ValueError("Transformer block_size must cover the sequence length")
    return merged