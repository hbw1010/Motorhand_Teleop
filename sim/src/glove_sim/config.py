from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SIM_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else SIM_ROOT / "config" / "default.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"配置不是对象: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    config["_sim_root"] = str(SIM_ROOT)
    for key, value in dict(config.get("paths", {})).items():
        raw = Path(value)
        config["paths"][key] = str((SIM_ROOT / raw).resolve() if not raw.is_absolute() else raw)
    return config


def path_of(config: dict[str, Any], key: str) -> Path:
    return Path(config["paths"][key])
