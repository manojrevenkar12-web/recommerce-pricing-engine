"""Market price per title: what a used copy in very good condition sells for.

Titles with competitor offers get a robust estimate from the feed: the median of the last 28 days,
switching to the last 7 days when they have moved (a new edition, a price war), so one dumping offer
does not move the price. Titles without offers (new releases, the long tail) get a cold-start model
on their attributes, with a P10-P90 range. The two are blended by how much feed there is.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from rpe.catalog import Catalog, title_features
from rpe.market import MarketView

COLD_FEATURES = ["category", "fmt", "english", "title_age", "days_since_release", "log_new_price", "log_rank"]
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40, "verbose": -1, "seed": 7}


@dataclass
class Snapshot:
    """What the market view says about every title at the end of `day`."""

    day: int
    feed: np.ndarray  # robust log price from the feed, NaN without feed
    n: np.ndarray  # feed observations in the last 28 days (0.5 = only an older price)
    feed_sd: np.ndarray
    rank: np.ndarray
    offers: np.ndarray
    ever_seen: np.ndarray
    last_price: np.ndarray
    quote: np.ndarray
    quote_age: np.ndarray
    own_sales: np.ndarray
    own_expo: np.ndarray


def snapshot(mv: MarketView, day: int) -> Snapshot:
    L28 = mv.recent_log_prices(28)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n28 = np.isfinite(L28).sum(axis=0)
        med28 = np.nanmedian(L28, axis=0) if len(L28) else np.full(mv.n, np.nan)
        sd = np.nanstd(L28, axis=0) if len(L28) else np.full(mv.n, np.nan)
        L7 = L28[:7]
        n7 = np.isfinite(L7).sum(axis=0)
        med7 = np.nanmedian(L7, axis=0) if len(L7) else np.full(mv.n, np.nan)
    moved = (n7 >= 3) & (np.abs(med7 - med28) > 0.25)
    est = np.where(moved, med7, med28)
    stale = ~np.isfinite(est) & np.isfinite(mv.last_price)  # nothing in 28 days but seen before
    est = np.where(stale, np.log(mv.last_price), est)
    n = np.where(stale, 0.5, n28).astype(float)
    return Snapshot(
        day,
        est,
        n,
        sd,
        mv.rank.copy(),
        mv.offers.copy(),
        mv.ever_seen.copy(),
        mv.last_price.copy(),
        mv.quote.copy(),
        day - mv.quote_day,
        mv.own_sales.copy(),
        mv.own_expo.copy(),
    )


@dataclass
class MarketEstimate:
    log_m: np.ndarray
    sd: np.ndarray
    n_obs: np.ndarray
    from_feed: np.ndarray
    log_p10: np.ndarray
    log_p90: np.ndarray

    @property
    def m(self) -> np.ndarray:
        return np.exp(self.log_m)


@dataclass
class MarketPriceModel:
    mid: lgb.Booster
    lo: lgb.Booster
    hi: lgb.Booster
    sigma_model: float
    sigma_feed: float

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> MarketPriceModel:
        """rows: COLD_FEATURES + `target` (robust log feed price of well-observed titles) + `feed_sd`."""
        X, y = rows[COLD_FEATURES], rows["target"].to_numpy()
        cats = ["category", "fmt"]
        ds = lgb.Dataset(X, y, categorical_feature=cats, free_raw_data=False)
        mid = lgb.train(PARAMS, ds, 400)
        lo = lgb.train({**PARAMS, "objective": "quantile", "alpha": 0.1}, ds, 300)
        hi = lgb.train({**PARAMS, "objective": "quantile", "alpha": 0.9}, ds, 300)
        # out-of-sample spread: hold out half of the titles
        title_half = rows["title"].to_numpy() % 2 == 0
        a = lgb.train(PARAMS, lgb.Dataset(X[title_half], y[title_half], categorical_feature=cats), 400)
        sigma_model = float(np.std(y[~title_half] - a.predict(X[~title_half])))
        return cls(mid, lo, hi, sigma_model, float(np.nanmedian(rows["feed_sd"])))

    def cold(self, cat: Catalog, snap: Snapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cold-start log price (P50, P10, P90) for every title from its attributes."""
        X = title_features(cat, snap, snap.day, np.full(cat.n, np.nan), snap.n)
        mid, lo, hi = (b.predict(X[COLD_FEATURES]) for b in (self.mid, self.lo, self.hi))
        return mid, lo, hi

    def estimate(self, cat: Catalog, snap: Snapshot, cold: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> MarketEstimate:
        cm, clo, chi = cold if cold is not None else self.cold(cat, snap)
        has = np.isfinite(snap.feed)
        n_eff = np.minimum(snap.n, 4.0)
        k = (self.sigma_feed / self.sigma_model) ** 2
        w = np.where(has, n_eff / (n_eff + k), 0.0)
        log_m = w * np.nan_to_num(snap.feed) + (1 - w) * cm
        sd = np.sqrt((w * self.sigma_feed) ** 2 / np.maximum(n_eff, 1) + ((1 - w) * self.sigma_model) ** 2)
        feed_based = w > 0.5
        return MarketEstimate(
            log_m,
            sd,
            snap.n,
            feed_based,
            np.where(feed_based, log_m - 1.28 * sd, np.minimum(clo, log_m)),
            np.where(feed_based, log_m + 1.28 * sd, np.maximum(chi, log_m)),
        )

    def estimate_live(
        self, cat: Catalog, mv: MarketView, day: int, cold: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> MarketEstimate:
        return self.estimate(cat, snapshot(mv, day), cold)


def training_rows(cat: Catalog, snaps: list[Snapshot]) -> pd.DataFrame:
    """Cold-start training rows: titles with at least 10 feed days in the 28 before each snapshot."""
    out = []
    for s in snaps:
        ok = np.isfinite(s.feed) & (s.n >= 10)
        X = title_features(cat, s, s.day, s.feed, s.n)
        X["target"], X["feed_sd"], X["title"] = s.feed, s.feed_sd, np.arange(cat.n)
        out.append(X[ok])
    return pd.concat(out, ignore_index=True)
