from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "tbrgs_defaults.json"


def _resolve_data_paths(cfg: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        return cfg

    for key in ("traffic_file", "site_file", "location_fallback_file"):
        value = data_cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        p = Path(value)
        if not p.is_absolute():
            data_cfg[key] = str((config_dir / p).resolve())
    return cfg


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    if config_path:
        path = Path(config_path)
        if not path.is_absolute() and not path.exists():
            path = (PROJECT_ROOT / path).resolve()
    else:
        path = DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return _resolve_data_paths(cfg, path.parent)
