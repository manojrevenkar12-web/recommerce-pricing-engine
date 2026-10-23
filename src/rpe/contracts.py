"""Data contracts for every raw export, and the ingest that enforces them.

Each source has a YAML contract in config/contracts (owner, key, typed columns with rules, references
to other sources). Ingest moves every row that breaks its contract to a quarantine table with the
reason and loads the rest. Rows whose key repeats keep the first occurrence. Nothing downstream
reads a source without a contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rpe import ean as ean13
from rpe.config import CLEAN, CONFIG, RAW

ORDER = [
    "catalogue",
    "opening_stock",
    "opening_in_transit",
    "baskets",
    "receipts",
    "offers",
    "market_feed",
    "buyback_quotes",
    "sales_ranks",
    "prices",
    "sales",
    "disposals",
    "warehouse_daily",
]


@dataclass
class Contract:
    source: str
    owner: str
    description: str
    key: list[str]
    columns: dict[str, dict[str, Any]]
    relations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Contract:
        d = yaml.safe_load(path.read_text())
        return cls(d["source"], d["owner"], d["description"], d.get("key") or [], d["columns"], d.get("relations") or [])


def load_contracts(directory: Path | None = None) -> dict[str, Contract]:
    directory = directory or CONFIG / "contracts"
    return {c.source: c for c in (Contract.load(p) for p in sorted(directory.glob("*.yaml")))}


@dataclass
class SourceReport:
    source: str
    owner: str
    rows_read: int = 0
    loaded: int = 0
    quarantined: int = 0
    reasons: dict[str, int] = field(default_factory=dict)


def _typed(s: pd.Series, kind: str) -> tuple[pd.Series, pd.Series]:
    """The column converted to its contract type, and a mask of values that could not be."""
    present = s.notna()
    if kind == "string":
        return s.astype("string"), pd.Series(False, index=s.index)
    if kind == "int":
        v = pd.to_numeric(s, errors="coerce")
        bad = present & (v.isna() | (v.fillna(0) % 1 != 0))
        return v.astype("Int64"), bad
    if kind == "float":
        v = pd.to_numeric(s, errors="coerce").astype(float)
        return v, present & ~np.isfinite(v.fillna(0.0))
    if kind == "bool":
        ok = s.isin([True, False]) | ~present
        return s.astype("boolean"), ~ok
    if kind == "date":
        v = pd.to_datetime(s, errors="coerce")
        return v, present & v.isna()
    raise ValueError(f"unknown type {kind}")


def check(df: pd.DataFrame, c: Contract, parents: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean rows, quarantined rows with a `reason`)."""
    df = df.copy()
    reason = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, why: str) -> None:
        nonlocal reason
        reason = reason.where(~(mask & (reason == "")), why)

    missing = [col for col in c.columns if col not in df.columns]
    if missing:
        raise ValueError(f"{c.source}: export lacks columns {missing}")
    for col, rule in c.columns.items():
        v, bad = _typed(df[col], rule["type"])
        df[col] = v
        flag(bad, f"{col}: not a {rule['type']}")
        if rule.get("required"):
            flag(v.isna(), f"{col}: missing")
        nn = v.notna()
        if "min" in rule:
            flag(nn & (v < rule["min"]), f"{col}: below {rule['min']}")
        if "min_exclusive" in rule:
            flag(nn & (v <= rule["min_exclusive"]), f"{col}: not above {rule['min_exclusive']}")
        if "max" in rule:
            flag(nn & (v > rule["max"]), f"{col}: above {rule['max']}")
        if "allowed" in rule:
            flag(nn & ~v.isin(rule["allowed"]), f"{col}: not one of the allowed values")
        if rule.get("rule") == "ean13":
            valid = {x: ean13.valid(str(x)) for x in v.dropna().unique()}
            flag(nn & ~v.map(valid).fillna(False).astype(bool), f"{col}: invalid EAN-13 check digit")
    for rel in c.relations:
        allowed: set = set()
        for ref in rel["references"]:
            t, col = ref.split(".")
            if t in parents:
                allowed |= set(parents[t][col].dropna().tolist())
        v = df[rel["column"]]
        flag(v.notna() & ~v.isin(allowed), f"{rel['column']}: unknown in {' or '.join(rel['references'])}")
    if c.key:
        ok = reason == ""
        dup = pd.Series(False, index=df.index)
        dup[ok] = df[ok].duplicated(c.key, keep="first")
        flag(dup, f"duplicate {'+'.join(c.key)}")
    bad = reason != ""
    q = df[bad].copy()
    q.insert(0, "reason", reason[bad])
    return df[~bad].reset_index(drop=True), q


def ingest(raw: Path = RAW, clean: Path = CLEAN) -> dict[str, Any]:
    contracts = load_contracts()
    unknown = sorted({p.stem for p in raw.glob("*.parquet")} - set(contracts))
    if unknown:
        raise ValueError(f"raw sources without a contract: {unknown}")
    clean.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, pd.DataFrame] = {}
    reports, quarantine = [], []
    for name in ORDER:
        c = contracts[name]
        df = pd.read_parquet(raw / f"{name}.parquet")
        good, bad = check(df, c, loaded)
        loaded[name] = good
        good.to_parquet(clean / f"{name}.parquet", index=False)
        r = SourceReport(name, c.owner, len(df), len(good), len(bad), bad.reason.value_counts().to_dict() if len(bad) else {})
        reports.append(r)
        if len(bad):
            quarantine.append(
                pd.DataFrame(
                    {
                        "source": name,
                        "reason": bad.reason,
                        "row": [json.dumps(r, default=str) for r in bad.drop(columns="reason").to_dict("records")],
                    }
                )
            )
    q = pd.concat(quarantine, ignore_index=True) if quarantine else pd.DataFrame(columns=["source", "reason", "row"])
    q.to_parquet(clean / "_quarantine.parquet", index=False)
    out = {"sources": [asdict(r) for r in reports], "quarantined": len(q)}
    (clean / "_ingest_report.json").write_text(json.dumps(out, indent=2))
    return out
