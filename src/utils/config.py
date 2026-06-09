"""Load and expose project configuration from configs/config.yaml."""

from __future__ import annotations
from pathlib import Path
import yaml


_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else _ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_groups(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else _ROOT / "configs" / "wc2026_groups.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# Singleton — imported everywhere else
CONFIG = load_config()
GROUPS = load_groups()
