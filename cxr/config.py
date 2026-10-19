from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = Path(cfg["data_dir"])
    cfg["output_dir"] = Path(cfg["output_dir"])
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg


def slug(name: str) -> str:
    return name.replace(" ", "_")
