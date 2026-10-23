"""The pricing service behind the API: the engine with the market, stock and warehouse as of the
latest data, and the request-level logic (explanations, unknown titles, the minimum basket)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from rpe.calendar import to_date
from rpe.catalog import CATEGORIES
from rpe.config import MODELS, WAREHOUSE, costs, load
from rpe.engine.core import Engine
from rpe.engine.rules import sell_bounds, validate
from rpe.market import MarketView
from rpe.policy import GRADES, Items, Warehouse
from rpe.warehouse import query, table

REASONS = {
    "buy": "We buy this.",
    "no_margin": "Not buying this right now: at any offer we could make, the expected resale would not cover handling and the risk it does not sell.",
    "slow_seller": "Not buying this right now: copies of this title are unlikely to sell within six months.",
    "warehouse_full": "Not buying anything right now: the warehouse is full.",
    "unknown_title": "We do not know this barcode.",
}


@dataclass
class StockState:
    stock: int
    in_transit: int
    capacity: int
    depth: np.ndarray


class PricingService:
    def __init__(self, bundle: Any, market: MarketView, stock: StockState, day: int, occupancy_history: list[int] | None = None):
        problems = validate(load("pricing"))
        if problems:
            raise ValueError("invalid config/pricing.yaml: " + "; ".join(problems))
        self.bundle = bundle
        self.cat = bundle.catalog
        self.engine = Engine(bundle, market)
        self.stock = stock
        self.day = day
        self.c = costs()
        self.lock = threading.Lock()
        for k, occ in enumerate(occupancy_history or []):
            self.engine.capacity.update(day - len(occupancy_history or []) + k, Warehouse(occ, 0, stock.capacity, stock.depth))
        self.engine.capacity.update(day, self.warehouse)
        with self.lock:
            self.engine.estimate(day)  # warm up

    @classmethod
    def from_warehouse(cls, models: Any = MODELS, wh: Any = WAREHOUSE) -> PricingService:
        from rpe.train import Bundle

        b = Bundle.load(models)
        last = int(query("select max(day) as d from warehouse_daily", wh).d[0])
        mv = MarketView.from_history(b.catalog.n, table("feed", wh), table("quotes", wh), table("ranks", wh), last, table("units", wh))
        u = query(f"select idx from units where coalesce(sold_day, 99999) > {last} and coalesce(disposed_day, 99999) > {last}", wh)
        daily = query("select * from warehouse_daily order by day", wh)
        lastrow = daily.iloc[-1]
        depth = np.bincount(u.idx.to_numpy(), minlength=b.catalog.n)
        st = StockState(int(lastrow.stock_units), int(lastrow.in_transit_units), int(lastrow.capacity_units), depth)
        occ = (daily.stock_units + daily.in_transit_units).tail(15).astype(int).tolist()[:-1]
        return cls(b, mv, st, last + 1, occ)

    @property
    def warehouse(self) -> Warehouse:
        return Warehouse(self.stock.stock, self.stock.in_transit, self.stock.capacity, self.stock.depth)

    # -- buying -----------------------------------------------------------------------------------
    def quote(self, eans: list[str], returning: bool = False) -> dict[str, Any]:
        idx = self.cat.index_of(np.array(eans, dtype=str))
        known = idx >= 0
        rows: list[dict[str, Any]] = [{} for _ in eans]
        if known.any():
            k = np.flatnonzero(known)
            items = Items(np.zeros(len(k), int), np.zeros(len(k), int), np.full(len(k), int(returning)), idx[k])
            with self.lock:
                d = self.engine.offer_details(self.day, items, self.warehouse)
            for j, row in zip(k, d.to_dict("records"), strict=True):
                rows[j] = _offer_row(eans[j], row)
        for j in np.flatnonzero(~known):
            rows[j] = {"ean": eans[j], "offer_eur": 0.0, "decision": "unknown_title", "message": REASONS["unknown_title"]}
        total = round(sum(r["offer_eur"] for r in rows), 2)
        mb = self.c["buy"]["min_basket_eur"]
        return {
            "as_of": to_date(self.day).isoformat(),
            "items": rows,
            "basket_total_eur": total,
            "min_basket_eur": mb,
            "meets_min_basket": total >= mb,
            "shelf_price_eur_per_day": round(self.engine.capacity.price, 4),
            "warehouse_fill": round((self.stock.stock + self.stock.in_transit) / self.stock.capacity, 4),
        }

    # -- selling ----------------------------------------------------------------------------------
    def prices(self, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
        eans = np.array([u["ean"] for u in units], dtype=str)
        idx = self.cat.index_of(eans)
        out: list[dict[str, Any]] = []
        ok = np.flatnonzero(idx >= 0)
        if len(ok):
            g = np.array([GRADES.index(units[i]["grade"]) for i in ok])
            listed = np.array([self.day - int(units[i].get("days_listed", 0)) for i in ok])
            cur = np.array([float(units[i]["current_price"]) if units[i].get("current_price") else np.nan for i in ok])
            with self.lock:
                age = self.day - listed
                depth = np.maximum(self.stock.depth[idx[ok]], 1)
                hold = self.c["stock"]["holding_per_unit_day"] + self.engine.capacity.price
                left = np.maximum(self.engine.cfg["sell"]["max_age_days"] - age, 7).astype(float)
                v = self.engine.value_units(self.day, idx[ok], g, age, depth, cur, left, hold)
                liq = self.engine.liquidation(age, v.value, self.stock.stock)
                est = self.engine.estimate(self.day)
            b = sell_bounds(self.engine.cfg, self.cat.category[idx[ok]], self.cat.fmt[idx[ok]], g)
            k = self.bundle.demand.frailty_shape
            for n, i in enumerate(ok):
                p90 = 1 - (1 + 90 * v.intensity[n] / k) ** (-k)
                out.append(
                    {
                        "unit_id": units[i]["unit_id"],
                        "ean": units[i]["ean"],
                        "price_eur": round(float(v.price[n]), 2),
                        "liquidate": bool(liq[n]),
                        "market_price_eur": round(float(np.exp(est.log_m[idx[i]])), 2),
                        "market_price_from": "competitor feed" if est.from_feed[idx[i]] else "attribute model (no competitor offers)",
                        "reference_price_eur": round(float(v.ref[n]), 2),
                        "allowed_range_eur": [round(float(v.lo[n]), 2), round(float(v.hi[n]), 2)],
                        "steering_rules": b.rules[n],
                        "sellability_90d": round(float(p90), 3),
                        "expected_days_listed": round(float(v.days_listed[n]), 1),
                        "expected_value_eur": round(float(v.value[n]), 2),
                    }
                )
        for i in np.flatnonzero(idx < 0):
            out.append({"unit_id": units[i]["unit_id"], "ean": units[i]["ean"], "error": "unknown_title"})
        return out

    # -- titles -----------------------------------------------------------------------------------
    def title(self, ean: str) -> dict[str, Any] | None:
        i = int(self.cat.index_of(np.array([ean]))[0])
        if i < 0:
            return None
        with self.lock:
            est = self.engine.estimate(self.day)
            X = self.engine.features(self.day, np.full(4, i), np.zeros(4), est)
            depth = np.full(4, max(int(self.stock.depth[i]), 1))
            sell = self.bundle.demand.sellability(X, np.arange(4), depth, 90)
        mv = self.engine.mv
        return {
            "ean": ean,
            "category": CATEGORIES[self.cat.category[i]],
            "market_price_eur": round(float(np.exp(est.log_m[i])), 2),
            "market_price_p10_p90_eur": [round(float(np.exp(est.log_p10[i])), 2), round(float(np.exp(est.log_p90[i])), 2)],
            "market_price_from": "competitor feed" if est.from_feed[i] else "attribute model (no competitor offers)",
            "competitor_lowest_eur": None if not np.isfinite(mv.last_price[i]) else round(float(mv.last_price[i]), 2),
            "competitor_buyback_quote_eur": None if not np.isfinite(mv.quote[i]) else round(float(mv.quote[i]), 2),
            "sales_rank": None if not np.isfinite(mv.rank[i]) else int(mv.rank[i]),
            "copies_in_stock": int(self.stock.depth[i]),
            "sellability_90d_by_grade": {g: round(float(p), 3) for g, p in zip(GRADES, sell, strict=True)},
        }

    def info(self) -> dict[str, Any]:
        card = self.bundle.card
        trained = datetime.fromisoformat(card["trained_at"])
        return {
            "as_of": to_date(self.day).isoformat(),
            "models_trained_at": card["trained_at"],
            "model_age_days": round((datetime.now(UTC) - trained).total_seconds() / 86400, 2),
            "history": card["history"],
            "data_fingerprint": card["data_fingerprint"],
            "titles": int(self.cat.n),
            "stock_units": self.stock.stock,
            "warehouse_fill": round((self.stock.stock + self.stock.in_transit) / self.stock.capacity, 4),
        }


def _offer_row(ean: str, r: dict[str, Any]) -> dict[str, Any]:
    def f(x: Any, nd: int = 3) -> float | None:
        return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)

    return {
        "ean": ean,
        "offer_eur": round(float(r["offer"]), 2),
        "decision": r["reason"],
        "message": REASONS[r["reason"]],
        "p_seller_accepts": f(r["p_keep"]),
        "expected_profit_eur": f(r["expected_profit"], 2),
        "value_if_received_eur": f(r["value_if_received"], 2),
        "p_reject_on_grading": f(r["p_reject"]),
        "market_price_eur": f(r["market_price"], 2),
        "market_price_from": "competitor feed" if r["market_from_feed"] else "attribute model (no competitor offers)",
        "planned_list_price_very_good_eur": f(r["list_price_very_good"], 2),
        "sellability_90d": f(r["sellability_90d"]),
        "copies_in_stock_or_on_the_way": int(r["copies_in_stock"]),
        "competitor_buyback_quote_eur": f(r["competitor_quote"], 2),
    }


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)
