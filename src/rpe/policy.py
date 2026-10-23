"""What a pricing policy sees and decides. The simulator, the backtests and the API all speak this.

A policy only ever receives what the company itself would know: the catalogue, the competitor feed,
sales ranks, competitor buy-back quotes, its own stock, its own offers and its own sales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

GRADES = ["like_new", "very_good", "good", "acceptable"]
REJECT = 4


@dataclass
class MarketUpdate:
    day: int
    feed_idx: np.ndarray  # titles with a competitor lowest used price today
    feed_price: np.ndarray
    feed_offers: np.ndarray
    quote_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # competitor buy-back quotes (weekly)
    quote: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rank_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # sales ranks (monthly)
    rank: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sold_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # our sales yesterday
    sold_price: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Units:
    """Our stock as the warehouse system sees it."""

    unit_id: np.ndarray
    idx: np.ndarray  # title
    grade: np.ndarray  # 0..3
    listed_day: np.ndarray
    price: np.ndarray  # current price, NaN if never priced
    needs_price: np.ndarray  # new arrivals and the weekly repricing

    def __len__(self) -> int:
        return len(self.unit_id)


@dataclass
class Items:
    """Items scanned by sellers in the app today, one row per item."""

    basket: np.ndarray  # basket number within the day
    seller: np.ndarray
    visit: np.ndarray  # 0 for a seller's first basket
    idx: np.ndarray

    def __len__(self) -> int:
        return len(self.idx)


@dataclass
class Warehouse:
    stock: int
    in_transit: int
    capacity: int
    depth: np.ndarray  # units per title in stock or on the way


@dataclass
class PriceDecision:
    price: np.ndarray
    liquidate: np.ndarray
    reason: np.ndarray | None = None  # short code per unit, for the price log
    explore: np.ndarray | None = None  # multiplier when the unit is in a randomised price test, else NaN


@dataclass
class OfferDecision:
    offer: np.ndarray  # 0 = not buying
    explore: np.ndarray | None = None
    reason: np.ndarray | None = None


class Policy(Protocol):
    name: str

    def observe(self, update: MarketUpdate) -> None: ...

    def price(self, day: int, units: Units, wh: Warehouse) -> PriceDecision: ...

    def offer(self, day: int, items: Items, wh: Warehouse) -> OfferDecision: ...


class Composite:
    """Buy decisions from one policy, sell decisions from another (ablations and experiments)."""

    def __init__(self, buy: Policy, sell: Policy, name: str | None = None):
        self.buy, self.sell = buy, sell
        self.name = name or f"buy:{buy.name}+sell:{sell.name}"

    def observe(self, update: MarketUpdate) -> None:
        self.buy.observe(update)
        if self.sell is not self.buy:
            self.sell.observe(update)

    def price(self, day: int, units: Units, wh: Warehouse) -> PriceDecision:
        return self.sell.price(day, units, wh)

    def offer(self, day: int, items: Items, wh: Warehouse) -> OfferDecision:
        return self.buy.offer(day, items, wh)


class Split:
    """Two policies side by side, by experiment arm: sell side by title, buy side by seller."""

    def __init__(self, control: Policy, treatment: Policy, sell_arm: np.ndarray | None, buy_arm_of_seller=None, name: str = "split"):
        self.c, self.t = control, treatment
        self.sell_arm = sell_arm  # bool per title: True = treatment; None = control for all
        self.buy_arm_of_seller = buy_arm_of_seller  # callable seller ids -> bool, or None
        self.name = name

    def observe(self, update: MarketUpdate) -> None:
        self.c.observe(update)
        self.t.observe(update)

    def price(self, day: int, units: Units, wh: Warehouse) -> PriceDecision:
        dc = self.c.price(day, units, wh)
        if self.sell_arm is None:
            return dc
        dt = self.t.price(day, units, wh)
        arm = self.sell_arm[units.idx]
        return PriceDecision(np.where(arm, dt.price, dc.price), np.where(arm, dt.liquidate, dc.liquidate))

    def offer(self, day: int, items: Items, wh: Warehouse) -> OfferDecision:
        dc = self.c.offer(day, items, wh)
        if self.buy_arm_of_seller is None:
            return dc
        dt = self.t.offer(day, items, wh)
        arm = self.buy_arm_of_seller(items.seller)
        return OfferDecision(np.where(arm, dt.offer, dc.offer))
