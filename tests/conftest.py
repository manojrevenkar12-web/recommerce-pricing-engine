"""Pipeline tests need the generated history, the warehouse and trained models. They are built
once per session if missing (about four minutes); unit tests need nothing."""

from __future__ import annotations

import pytest

from rpe.config import CLEAN, MODELS, RAW, WAREHOUSE


@pytest.fixture(scope="session")
def built() -> None:
    if not (RAW / "offers.parquet").exists():
        from rpe.sim.history import generate

        generate()
    if not (CLEAN / "_ingest_report.json").exists():
        from rpe.contracts import ingest

        ingest()
    if not WAREHOUSE.exists():
        from rpe.warehouse import build

        build()
    if not (MODELS / "bundle.joblib").exists():
        from rpe.train import train

        train()


@pytest.fixture(scope="session")
def bundle(built):
    from rpe.train import Bundle

    return Bundle.load()


@pytest.fixture(scope="session")
def start(bundle):
    from rpe.evaluation.backtest import load_start

    return load_start(bundle)
