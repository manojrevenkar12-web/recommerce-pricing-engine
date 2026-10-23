"""Trains every model from the warehouse and writes them, with a model card, to models/."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from rpe.calendar import day, to_date
from rpe.catalog import CATEGORIES, Catalog
from rpe.config import CLEAN, MODELS, WAREHOUSE
from rpe.models.acceptance import AcceptanceModel, acceptance_rows
from rpe.models.demand import DemandModel, build_panel
from rpe.models.grading import GradingModel
from rpe.models.market_price import MarketPriceModel, training_rows
from rpe.models.replay import replay
from rpe.models.seller_value import SellerValueModel
from rpe.warehouse import query, table


class Bundle:
    """All fitted models plus the catalogue they were fitted on."""

    def __init__(
        self,
        catalog: Catalog,
        market: MarketPriceModel,
        demand: DemandModel,
        acceptance: AcceptanceModel,
        grading: GradingModel,
        seller: SellerValueModel,
        card: dict[str, Any],
    ):
        self.catalog, self.market, self.demand = catalog, market, demand
        self.acceptance, self.grading, self.seller = acceptance, grading, seller
        self.card = card

    def save(self, directory: Path = MODELS) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "bundle.joblib", compress=3)
        (directory / "model_card.json").write_text(json.dumps(self.card, indent=2, default=str))

    @staticmethod
    def load(directory: Path = MODELS) -> Bundle:
        return joblib.load(directory / "bundle.joblib")


def history_window(wh: Path = WAREHOUSE) -> tuple[int, int]:
    r = query("select min(day) as a, max(day) as b from warehouse_daily", wh)
    return int(r.a[0]), int(r.b[0])


def train(wh: Path = WAREHOUSE, out: Path = MODELS) -> Bundle:
    t0 = time.time()
    cat = Catalog.from_frame(table("dim_title", wh))
    first, last = history_window(wh)
    mondays = {d for d in range(first, last + 1) if to_date(d).weekday() == 0} | {last}
    feed, quotes, ranks = table("feed", wh), table("quotes", wh), table("ranks", wh)
    units = table("units", wh)
    snaps, _ = replay(cat.n, feed, quotes, ranks, first, last, mondays, units)
    timings = {"replay_s": round(time.time() - t0, 1)}

    # market price (cold start) on snapshots after the first month, when the 28-day window is full
    t = time.time()
    full = [snaps[d] for d in sorted(snaps) if d >= first + 28]
    market = MarketPriceModel.fit(training_rows(cat, full[::2]))
    timings["market_s"] = round(time.time() - t, 1)

    # demand and price response
    t = time.time()
    prices = pd.concat([_opening_prices(), table("prices", wh)], ignore_index=True)
    panel = build_panel(cat, units, prices, {d: s for d, s in snaps.items() if d >= first + 28}, market, last)
    demand = DemandModel.fit(panel)
    timings["demand_s"] = round(time.time() - t, 1)

    # sellers: will they keep an item at this offer, what grade arrives, will they come back
    t = time.time()
    offers = query("select o.*, b.visit, b.items from offers o join baskets b using (basket_id)", wh)
    acc_rows = acceptance_rows(cat, offers, snaps, market)
    acceptance = AcceptanceModel.fit(acc_rows)
    grading = GradingModel.fit(cat, table("receipts", wh))
    seller = SellerValueModel.fit(table("baskets", wh), offers, acc_rows, units, last)
    timings["sellers_s"] = round(time.time() - t, 1)

    parcel = query(
        "select avg(k) as k from (select basket_id, count(*) filter (where kept) as k from offers group by 1) join baskets using (basket_id) where sent",
        wh,
    )
    card = {
        "items_per_parcel": round(float(parcel.k[0]), 2),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": [to_date(first).isoformat(), to_date(last).isoformat()],
        "data_fingerprint": _fingerprint(wh),
        "rows": {
            "market_snapshots": len(full),
            "demand_unit_weeks": len(panel),
            "demand_test_rows": int(panel.test.sum()),
            "offers": len(acc_rows),
        },
        "price_response": {
            c: {"beta": round(float(demand.beta[i]), 3), "alpha": round(float(demand.alpha[i]), 3)} for i, c in enumerate(CATEGORIES)
        },
        "condition_factors": dict(
            zip(["like_new", "very_good", "good", "acceptable"], np.exp(demand.log_q).round(3).tolist(), strict=True)
        ),
        "market_sigma_model": round(market.sigma_model, 3),
        "market_sigma_feed": round(market.sigma_feed, 3),
        "acceptance": acceptance.card(),
        "seller_value": seller.card(),
        "demand_fit": demand.history,
        "timings": timings,
    }
    b = Bundle(cat, market, demand, acceptance, grading, seller, card)
    b.save(out)
    return b


def _opening_prices() -> pd.DataFrame:
    """Prices of the opening stock as they stood when the export starts."""
    o = pd.read_parquet(CLEAN / "opening_stock.parquet")
    return pd.DataFrame({"day": [day(d) for d in o.listed_date], "unit_id": o.unit_id, "price": o.price_eur, "test_multiplier": np.nan})


def _fingerprint(wh: Path) -> str:
    r = query("select count(*) as n, sum(price) as s from feed", wh)
    u = query("select count(*) as n, sum(coalesce(sold_price, 0)) as s from units", wh)
    return hashlib.sha256(f"{r.n[0]}:{r.s[0]:.2f}:{u.n[0]}:{u.s[0]:.2f}".encode()).hexdigest()[:12]
