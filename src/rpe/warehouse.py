"""Builds the analytics warehouse (DuckDB) from the clean, contract-checked exports."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd

from rpe.calendar import EPOCH
from rpe.config import CLEAN, WAREHOUSE


def build(clean: Path = CLEAN, path: Path = WAREHOUSE) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        for f in sorted(resources.files("rpe.sql").iterdir(), key=lambda p: p.name):
            if f.name.endswith(".sql"):
                con.execute(f.read_text().format(clean=clean.as_posix(), epoch=EPOCH.isoformat())).fetchall()
        tables = [r[0] for r in con.execute("select table_name from information_schema.tables order by 1").fetchall()]
        return {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in tables}  # type: ignore[index]
    finally:
        con.close()


def query(sql: str, path: Path = WAREHOUSE) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def table(name: str, path: Path = WAREHOUSE) -> pd.DataFrame:
    return query(f"select * from {name}", path)
