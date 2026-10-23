"""Process-pool worker for backtests. Kept free of heavy imports at module level so the thread
settings apply before LightGBM loads in a spawned process."""

from __future__ import annotations

import os
from typing import Any

_START: Any = None


def init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    global _START
    from rpe.evaluation.backtest import load_start
    from rpe.train import Bundle

    _START = load_start(Bundle.load())


def run(spec: dict[str, Any]) -> dict[str, Any]:
    from rpe.evaluation.backtest import run_one

    return run_one(_START, **spec)
