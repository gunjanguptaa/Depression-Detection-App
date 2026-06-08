from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

try:
    import tensorflow as tf
except Exception:  # TensorFlow is imported only when available in the runtime.
    tf = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load YAML config and resolve relative paths from the project root."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                cfg["paths"][key] = str(ROOT_DIR / p)
    return cfg


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for key in ["model_dir", "plots_dir", "cache_dir", "reports_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def save_preprocess_config(cfg: Dict[str, Any], output_path: str | Path | None = None) -> None:
    output_path = Path(output_path or cfg["paths"]["preprocess_config"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "features": cfg["features"],
        "training": {"threshold": cfg["training"].get("threshold", 0.5)},
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
