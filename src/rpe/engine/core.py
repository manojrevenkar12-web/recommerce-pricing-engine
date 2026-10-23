"""The pricing engine: buy offers, buy/no-buy, sell prices and liquidation, from the models.

Selling. For each unit the engine chooses the price that maximises its expected value over the
time it may still stay in stock:

    V(p) = P(sold)(p) * net(p) - (holding + shelf price) * E[days listed](p) + P(not sold)(p) * liquidation

with sale intensity from the demand model. It reprices weekly, so as a unit ages (fewer days left,
and the model learns it is a slow seller) the optimum falls: markdowns come out of the economics,
not out of a fixed age table. A unit is liquidated when keeping it is worth less than selling it in
bulk now.

Buying. For each scanned item it values a fresh unit in every condition it might arrive in (with the
copies already in stock or on the way), subtracts handling, parcel cost and the chance of a reject,
and chooses the offer that maximises

    P(seller keeps it | offer) * (expected value - offer) + long-term value of a more generous offer

or makes no offer when nothing clears the minimum profit or the title is unlikely to sell.

Warehouse. A shelf slot gets a price (EUR per day) that rises when projected fill exceeds its target
and falls when it does not; it is added to holding cost on both sides, so a full warehouse buys only
fast sellers and clears slow stock.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammainc

from rpe.catalog import title_features
from rpe.config import costs as load_costs
from rpe.config import load
from rpe.engine.rules import buy_bounds, price_interval, sell_bounds
from rpe.market import MarketView
from rpe.models.market_price import MarketEstimate
from rpe.policy import Items, MarketUpdate, OfferDecision, PriceDecision, Units, Warehouse


@dataclass
class Flags:
    """Engine parts that can be switched off, for ablations ("what is each part worth?")."""

    market_model: bool = True
    capacity: bool = True
    long_term: bool = True


def net_sale(p: np.ndarray, c: dict[str, Any]) -> np.ndarray:
    return p * (1 - c["sell"]["fee_rate"]) - c["sell"]["fee_fixed"] - c["sell"]["pick_pack"]


def hold_value(
    p: np.ndarray, mu: np.ndarray, days: np.ndarray, hold: float | np.ndarray, liq: float, c: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(expected value, P(sold within `days`), expected days listed) at constant price p and a
    constant sale intensity mu per day."""
    mu = np.maximum(mu, 1e-9)
    ps = -np.expm1(-mu * days)
    t = ps / mu
    return net_sale(p, c) * ps - hold * t + liq * (1 - ps), ps, t


def queue_value(
    p: np.ndarray, title_rate: np.ndarray, position: np.ndarray, days: np.ndarray, hold: float | np.ndarray, liq: float, c: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same for one more copy of a title we already hold: buyers of the title take one copy
    each, so an extra copy adds no sales now; it sells when the `position - 1` copies ahead of it
    have. With title sales at rate L, the k-th copy sells at a Gamma(k, L) time:
    P(sold by W) = P(Poisson(LW) >= k) and E[min(T, W)] = k/L P(Poisson(LW) >= k+1) + W P(Poisson(LW) < k)."""
    L = np.maximum(title_rate, 1e-9)
    x = L * days
    k = np.broadcast_to(position, x.shape).astype(float)
    ps = gammainc(k, x)
    t = k / L * gammainc(k + 1, x) + days * (1 - ps)
    return net_sale(p, c) * ps - hold * t + liq * (1 - ps), ps, t


class CapacityController:
    """Price of a shelf slot per day, steered by projected warehouse fill."""

    def __init__(self, cfg: dict[str, Any], enabled: bool = True):
        self.cfg, self.enabled = cfg, enabled
        self.price = float(cfg["start_price"])
        self.day = -1
        self.occ: deque[float] = deque(maxlen=cfg["lookahead_days"] + 1)
        self.projected = 0.0
        self.trend = 0.0  # smoothed daily change of occupied slots

    def update(self, day: int, wh: Warehouse) -> float:
        if day == self.day:
            return self.price
        self.day = day
        occ = wh.stock + wh.in_transit
        if self.occ:
            self.trend = 0.9 * self.trend + 0.1 * (occ - self.occ[-1])
        self.occ.append(occ)
        self.projected = (occ + self.trend * self.cfg["lookahead_days"]) / wh.capacity
        if self.enabled:
            gap = (self.projected - self.cfg["target_fill"]) / 0.01
            self.price = float(np.clip(self.price + self.cfg["step"] * gap, 0.0, self.cfg["max_price"]))
        else:
            self.price = 0.0
        return self.price


@dataclass
class Valuation:
    price: np.ndarray
    value: np.ndarray
    p_sold: np.ndarray
    days_listed: np.ndarray
    intensity: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    log_market: np.ndarray
    ref: np.ndarray


@dataclass
class Engine:
    bundle: Any
    mv: MarketView
    flags: Flags = field(default_factory=Flags)
    cfg: dict[str, Any] = field(default_factory=lambda: load("pricing"))
    c: dict[str, Any] = field(default_factory=load_costs)
    name: str = "engine"
    audit: bool = False  # keep every price decision with its reference price, for the guardrail check

    def __post_init__(self) -> None:
        self.audit_log: list[pd.DataFrame] = []
        self.cat = self.bundle.catalog
        self.d = self.bundle.demand
        self.q = np.exp(self.d.log_q)
        self.capacity = CapacityController(self.cfg["capacity"], self.flags.capacity)
        self._est: tuple[int, MarketEstimate] | None = None
        self._cold: tuple[int, Any] | None = None
        self.liq = self.c["stock"]["liquidation_value"] - self.c["stock"]["liquidation_handling"]
        self.parcel_items = float(self.bundle.card.get("items_per_parcel", 15.0))
        self.theta = self.bundle.seller.marginal_value() * self.cfg["buy"]["long_term_weight"] if self.flags.long_term else 0.0
        g = self.cfg["sell"]["price_grid"]
        self.n_grid = int(g[2])
        self.quad = self.d.quadrature(4)

    # -- market -----------------------------------------------------------------------------------
    def observe(self, update: MarketUpdate) -> None:
        self.mv.observe(update)
        self._est = None

    def estimate(self, day: int) -> MarketEstimate:
        if self._est is not None and self._est[0] == day:
            return self._est[1]
        # attributes and ranks change slowly: refresh the cold-start prices weekly
        if self._cold is None or day - self._cold[0] >= 7:
            from rpe.models.market_price import snapshot

            self._cold = (day, self.bundle.market.cold(self.cat, snapshot(self.mv, day)))
        est = self.bundle.market.estimate_live(self.cat, self.mv, day, self._cold[1])
        if not self.flags.market_model:
            # ablation: the raw last competitor price, the list-price rule where there is none
            raw = self.mv.last_price
            fallback = np.where(np.isfinite(self.cat.new_price), self.cat.new_price * 0.4, 3.99)
            est = MarketEstimate(
                np.log(np.where(np.isfinite(raw), raw, fallback)), est.sd, est.n_obs, np.isfinite(raw), est.log_p10, est.log_p90
            )
        self._est = (day, est)
        return est

    def features(self, day: int, idx: np.ndarray, days_in_stock: np.ndarray, est: MarketEstimate) -> pd.DataFrame:
        X = title_features(self.cat, self.mv, day, est.log_m, est.n_obs, idx)
        X["days_in_stock"] = days_in_stock
        return X

    # -- valuation --------------------------------------------------------------------------------
    def value_units(
        self,
        day: int,
        idx: np.ndarray,
        grade: np.ndarray,
        days_in_stock: np.ndarray,
        depth: np.ndarray,
        old: np.ndarray,
        days_left: np.ndarray,
        hold: float,
        marginal: bool = False,
    ) -> Valuation:
        """Best price and its value for each unit. `marginal`: value an extra copy behind the
        `depth - 1` we hold (buying), instead of one of `depth` copies sharing the demand (selling)."""
        est = self.estimate(day)
        X = self.features(day, idx, days_in_stock, est)
        log_m = est.log_m[idx]
        ref = np.exp(log_m) * self.q[grade]
        bounds = sell_bounds(self.cfg, self.cat.category[idx], self.cat.fmt[idx], grade)
        lo, hi = price_interval(self.cfg, self.c, ref, old, self.cat.new_price[idx], bounds)
        t = np.linspace(0, 1, self.n_grid)
        grid = np.exp(np.log(lo)[:, None] + (np.log(hi) - np.log(lo))[:, None] * t[None, :])
        grid = np.round(grid, 2)
        rate = self.d.unit_rate(X, depth)
        cat = self.cat.category[idx]
        r = np.log(grid) - log_m[:, None] - self.d.log_q[grade][:, None]
        mu = rate[:, None] * self.d.share(r, cat[:, None])
        # what waiting without a sale says about this copy: cumulative expected sales so far
        listed_before = np.isfinite(old) & (old > 0)
        r_now = np.log(np.where(listed_before, old, ref)) - log_m - self.d.log_q[grade]
        exposure = np.where(listed_before, rate * self.d.share(r_now, cat) * days_in_stock, 0.0)
        post = self.d.frailty_shape + exposure
        V = np.zeros_like(grid)
        ps = np.zeros_like(grid)
        tl = np.zeros_like(grid)
        for x, wq in zip(*self.quad, strict=True):
            z = (x / post)[:, None]
            if marginal:
                k = np.maximum(depth, 1).astype(float)[:, None]
                v_, p_, t_ = queue_value(grid, mu * z * k, k, days_left[:, None], hold, self.liq, self.c)
            else:
                v_, p_, t_ = hold_value(grid, mu * z, days_left[:, None], hold, self.liq, self.c)
            V += wq * v_
            ps += wq * p_
            tl += wq * t_
        j = V.argmax(axis=1)
        rows = np.arange(len(idx))
        return Valuation(grid[rows, j], V[rows, j], ps[rows, j], tl[rows, j], mu[rows, j] * self.d.frailty_shape / post, lo, hi, log_m, ref)

    # -- selling ----------------------------------------------------------------------------------
    def price(self, day: int, units: Units, wh: Warehouse) -> PriceDecision:
        self.mv.observe_stock(day, np.bincount(units.idx, minlength=self.cat.n))
        shelf = self.capacity.update(day, wh)
        hold = self.c["stock"]["holding_per_unit_day"] + shelf
        n = len(units)
        price = units.price.copy()
        liquidate = np.zeros(n, bool)
        need = np.flatnonzero(units.needs_price)
        if len(need) == 0:
            return PriceDecision(price, liquidate)
        age = day - units.listed_day[need]
        depth = np.bincount(units.idx, minlength=self.cat.n)[units.idx[need]]
        s = self.cfg["sell"]
        left = np.maximum(s["max_age_days"] - age, 7).astype(float)
        v = self.value_units(day, units.idx[need], units.grade[need], age, depth, units.price[need], left, hold)
        price[need] = v.price
        if self.audit:
            self.audit_log.append(
                pd.DataFrame(
                    {
                        "day": day,
                        "idx": units.idx[need],
                        "grade": units.grade[need],
                        "old": units.price[need],
                        "price": v.price,
                        "ref": v.ref,
                    }
                )
            )
        liquidate[need] = self.liquidation(age, v.value, n)
        return PriceDecision(price, liquidate)

    def liquidation(self, age: np.ndarray, value: np.ndarray, stock_units: int) -> np.ndarray:
        """Liquidate when keeping a copy is worth less than selling it in bulk now, least valuable
        first and at most a set share of the stock per week; always at the end of its shelf life."""
        s = self.cfg["sell"]
        want = (age >= s["liquidate_min_age_days"]) & (value < self.liq - s["liquidate_margin_eur"])
        cap = int(s["max_liquidation_share_per_week"] * stock_units)
        if want.sum() > cap:
            order = np.argsort(np.where(want, value, np.inf))
            keep = np.zeros(len(want), bool)
            keep[order[:cap]] = True
            want &= keep
        return want | (age >= s["max_age_days"])

    # -- buying -----------------------------------------------------------------------------------
    def offer_details(self, day: int, items: Items, wh: Warehouse) -> pd.DataFrame:
        shelf = self.capacity.update(day, wh)
        hold = self.c["stock"]["holding_per_unit_day"] + shelf
        b = self.cfg["buy"]
        idx = items.idx
        n = len(idx)
        est = self.estimate(day)
        # copies of this title in stock or on the way, plus the ones scanned earlier today
        order = np.argsort(items.basket, kind="stable")
        dup = np.zeros(n)
        seen: dict[int, int] = {}
        for k in order:
            dup[k] = seen.get(int(idx[k]), 0)
            seen[int(idx[k])] = int(dup[k]) + 1
        depth = wh.depth[idx] + dup + 1
        pg = self.bundle.grading.probs(self.cat, idx, day)
        vals = np.zeros((n, 4))
        sell90 = np.zeros(n)
        for g in range(4):
            v = self.value_units(
                day,
                idx,
                np.full(n, g),
                np.zeros(n),
                depth,
                np.full(n, np.nan),
                np.full(n, float(self.cfg["sell"]["max_age_days"])),
                hold,
                marginal=True,
            )
            vals[:, g] = v.value
            if g == 1:
                k = self.d.frailty_shape
                sell90 = 1 - (1 + 90 * v.intensity / k) ** (-k)
                sell180 = 1 - (1 + 180 * v.intensity / k) ** (-k)
                list_price = v.price
        p_ok = pg[:, :4].sum(axis=1)
        v_ok = (pg[:, :4] * vals).sum(axis=1) / np.maximum(p_ok, 1e-9)
        fixed = (
            self.c["buy"]["inbound_handling"]
            + self.c["buy"]["basket_shipping"] / self.parcel_items
            + pg[:, 4] * self.c["buy"]["reject_disposal"]
        )
        # offer grid, capped by policy
        lo, hi, k = self.cfg["buy"]["offer_grid"]
        grid = np.exp(np.linspace(np.log(lo), np.log(hi), int(k)))
        bounds = buy_bounds(self.cfg, self.cat.category[idx], self.cat.fmt[idx])
        cap = bounds.max_offer_share * np.exp(est.log_m[idx])
        G = np.minimum(grid[None, :], np.maximum(cap, lo)[:, None])
        log_m = est.log_m[idx]
        quote, qage = self.mv.quote[idx], day - self.mv.quote_day[idx]
        returning = (items.visit > 0).astype(float)
        pacc = self.bundle.acceptance.prob(G, log_m, quote, qage, self.cat.category[idx], returning)
        margin = p_ok[:, None] * (v_ok[:, None] - G) - fixed[:, None]
        obj = pacc * margin
        j = obj.argmax(axis=1)
        rows = np.arange(n)
        buy = obj[rows, j] > b["min_expected_profit"]
        offer = G[rows, j]
        if self.theta > 0 and buy.any():
            # long-term value of a more generous basket (only for items worth buying on their own):
            # d log(satisfaction) / d offer = 1 / basket offer total
            tot = np.bincount(items.basket, weights=np.where(buy, offer, 0.0), minlength=int(items.basket.max()) + 1)[items.basket]
            lt = obj + self.theta * G / np.maximum(tot, 1.0)[:, None]
            j = np.where(buy, lt.argmax(axis=1), j)
            offer = G[rows, j]
        reason = np.full(n, "buy", dtype=object)
        reason[~buy] = "no_margin"
        slow = sell180 < b["min_sellability_180d"]
        reason[slow & buy] = "slow_seller"
        stop = (wh.stock + wh.in_transit) / wh.capacity >= b["emergency_stop_fill"]
        if stop:
            reason[:] = "warehouse_full"
        offer = np.where(reason == "buy", np.round(offer, 2), 0.0)
        return pd.DataFrame(
            {
                "idx": idx,
                "offer": offer,
                "reason": reason,
                "p_keep": np.where(reason == "buy", pacc[rows, j], np.nan),
                "expected_profit": np.where(reason == "buy", (pacc * margin)[rows, j], np.nan),
                "value_if_received": v_ok,
                "p_reject": pg[:, 4],
                "market_price": np.exp(log_m),
                "market_from_feed": est.from_feed[idx],
                "list_price_very_good": list_price,
                "sellability_90d": sell90,
                "copies_in_stock": depth - 1,
                "shelf_price_per_day": shelf,
                "competitor_quote": quote,
            }
        )

    def offer(self, day: int, items: Items, wh: Warehouse) -> OfferDecision:
        d = self.offer_details(day, items, wh)
        return OfferDecision(d.offer.to_numpy(), None, d.reason.to_numpy())
