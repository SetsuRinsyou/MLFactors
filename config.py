from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent / "config"

# 各专项配置文件路径
_SELECTION_CONFIG_PATH = _CONFIG_DIR / "selection.yaml"
_TIMING_CONFIG_PATH    = _CONFIG_DIR / "timing.yaml"

# 向后兼容：get_config() / load_config() 默认加载选股配置
_DEFAULT_CONFIG_PATH = _SELECTION_CONFIG_PATH

_global_config: dict[str, Any] | None = None
_global_timing_config: dict[str, Any] | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml(default_path: Path, user_config_path: str | Path | None) -> dict[str, Any]:
    with open(default_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if user_config_path is not None:
        with open(user_config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    return config


# ──────────────────────────────────────────────────────────────────────
#  选股类配置
# ──────────────────────────────────────────────────────────────────────

def load_config(user_config_path: str | Path | None = None) -> dict[str, Any]:
    """加载选股类因子测试配置（基于 config/selection.yaml）。"""
    global _global_config
    _global_config = _load_yaml(_SELECTION_CONFIG_PATH, user_config_path)
    return _global_config


def get_config() -> dict[str, Any]:
    """获取选股类因子测试配置（单例，首次调用时自动加载）。"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


# ──────────────────────────────────────────────────────────────────────
#  择时类配置
# ──────────────────────────────────────────────────────────────────────

def load_timing_config(user_config_path: str | Path | None = None) -> dict[str, Any]:
    """加载择时类因子测试配置（基于 config/timing.yaml）。"""
    global _global_timing_config
    _global_timing_config = _load_yaml(_TIMING_CONFIG_PATH, user_config_path)
    return _global_timing_config


def get_timing_config() -> dict[str, Any]:
    """获取择时类因子测试配置（单例，首次调用时自动加载）。"""
    global _global_timing_config
    if _global_timing_config is None:
        _global_timing_config = load_timing_config()
    return _global_timing_config

