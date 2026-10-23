"""Paths and configuration. Business thresholds live in config/*.yaml, never in code."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("RPE_ROOT", Path(__file__).resolve().parents[2]))
CONFIG = ROOT / "config"
DATA = Path(os.environ.get("RPE_DATA", ROOT / "data"))
RAW = DATA / "raw"
CLEAN = DATA / "clean"
TRUTH = DATA / "truth"
MODELS = Path(os.environ.get("RPE_MODELS", ROOT / "models"))
REPORTS = Path(os.environ.get("RPE_REPORTS", ROOT / "reports"))
WAREHOUSE = DATA / "warehouse.duckdb"


@lru_cache
def load(name: str) -> Any:
    """config/<name>.yaml (cached; callers must not mutate it)."""
    return yaml.safe_load((CONFIG / f"{name}.yaml").read_text())


def costs() -> dict[str, Any]:
    return load("costs")
