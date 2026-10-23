"""The current rule-based pricing (config/baseline.yaml): the control in every comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from rpe.config import costs, load
from rpe.market import MarketView
from rpe.policy import GRADES, Items, MarketUpdate, OfferDecision, PriceDecision, Units, Warehouse


def hash_uniform(*keys: np.ndarray | int) -> np.ndarray:
    """A deterministic uniform(0, 1) per element from integer keys (splitmix64)."""
    x = np.zeros(np.broadcast(*[np.asarray(k) for k in keys]).shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in keys:
            x = x ^ (np.asarray(k).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15) + (x << np.uint64(6)) + (x >> np.uint64(2)))
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


class Baseline:
    name = "baseline"

    def __init__(self, new_price: np.ndarray, market: MarketView, explore: bool = False, cfg: dict[str, Any] | None = None, salt: int = 7):
        self.cfg = cfg or load("baseline")
        self.c = costs()
        self.new_price = new_price
        self.mv = market
        self.explore = explore
        self.salt = salt
        self.cond = np.array([self.cfg["sell"]["condition_factor"][g] for g in GRADES])

    def observe(self, update: MarketUpdate) -> None:
        self.mv.observe(update)

    # -- buying -----------------------------------------------------------------------------------
    def offer(self, day: int, items: Items, wh: Warehouse) -> OfferDecision:
        b = self.cfg["buy"]
        if (wh.stock + wh.in_transit) / wh.capacity >= b["stop_buying_fill"]:
            return OfferDecision(np.zeros(len(items)), np.full(len(items), np.nan))
        rank = self.mv.rank[items.idx]
        lowest = self.mv.last_price[items.idx]
        bucket = np.searchsorted(np.array(b["rank_buckets"]), np.nan_to_num(rank, nan=1e12))
        rates = np.array([*b["rates"], 0.0])[bucket]
        offer = np.where(np.isfinite(lowest), rates * lowest - b["deduction"], 0.0)
        flat = ~np.isfinite(lowest) & (np.nan_to_num(rank, nan=1e12) <= b["no_feed_max_rank"])
        offer = np.where(flat, b["no_feed_offer"], offer)
        offer = np.where(offer >= self.c["buy"]["min_offer"], np.round(offer, 2), 0.0)
        explore = np.full(len(items), np.nan)
        if self.explore:
            e = self.cfg["exploration"]
            u = hash_uniform(items.seller, items.idx, day, self.salt)
            on = (u < e["offer_share"]) & (offer > 0)
            lo, hi = np.log(e["offer_multiplier"])
            mult = np.exp(lo + (hi - lo) * hash_uniform(items.seller, items.idx, day, self.salt + 1))
            explore = np.where(on, mult, np.nan)
            offer = np.where(on, np.maximum(np.round(offer * mult, 2), self.c["buy"]["min_offer"]), offer)
        return OfferDecision(offer, explore)

    # -- selling ----------------------------------------------------------------------------------
    def list_price(self, idx: np.ndarray, grade: np.ndarray, age: np.ndarray) -> np.ndarray:
        s = self.cfg["sell"]
        lowest = self.mv.last_price[idx]
        np_ = self.new_price[idx]
        fallback = np.where(np.isfinite(np_), np_ * s["no_feed_ratio"], s["no_feed_default"])
        p = np.where(np.isfinite(lowest), lowest * self.cond[grade] - s["undercut"], fallback * self.cond[grade])
        mult = np.ones(len(idx))
        for days, m in s["markdowns"]:
            mult = np.where(age >= days, m, mult)
        return np.maximum(p * mult, self.c["sell"]["min_price"])

    def price(self, day: int, units: Units, wh: Warehouse) -> PriceDecision:
        s = self.cfg["sell"]
        age = day - units.listed_day
        p = self.list_price(units.idx, units.grade, age)
        explore = np.full(len(units), np.nan)
        if self.explore:
            e = self.cfg["exploration"]
            period = day // e["sell_period_days"]
            on = hash_uniform(units.idx, period, self.salt + 2) < e["sell_share"]
            lo, hi = np.log(e["sell_multiplier"])
            mult = np.exp(lo + (hi - lo) * hash_uniform(units.idx, period, self.salt + 3))
            explore = np.where(on, mult, np.nan)
            p = np.where(on, np.maximum(p * mult, self.c["sell"]["min_price"]), p)
        liquidate = age >= s["liquidate_after_days"]
        return PriceDecision(np.round(p, 2), liquidate, None, explore)
