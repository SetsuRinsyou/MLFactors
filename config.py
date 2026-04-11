from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.yaml"

_global_config: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(user_config_path: str | Path | None = None) -> dict[str, Any]:
    global _global_config

    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if user_config_path is not None:
        with open(user_config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    _global_config = config
    return config


def get_config() -> dict[str, Any]:
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config
