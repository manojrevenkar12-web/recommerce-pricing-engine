"""Seller acceptance: will a seller keep an item in the basket at this offer?

A logistic model of the offer against what the main competitor would pay (its scraped buy-back
quote, or the category's typical quote-to-market ratio where there is no quote). Offers moved by the
randomised offer tests identify the slope; without them a high offer and a valuable title would be
indistinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rpe.catalog import CATEGORIES, Catalog
from rpe.models.market_price import MarketPriceModel, Snapshot

QUOTE_MAX_AGE = 60


def benchmark(
    log_m: np.ndarray, quote: np.ndarray, quote_age: np.ndarray, category: np.ndarray, kappa: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(log competitor buy-back benchmark, has a recent quote)."""
    has = np.isfinite(quote) & (quote_age <= QUOTE_MAX_AGE)
    return np.where(has, np.log(np.where(has, quote, 1.0)), log_m + np.log(kappa[category])), has


def design(
    log_offer: np.ndarray, log_b: np.ndarray, has_quote: np.ndarray, log_m: np.ndarray, category: np.ndarray, returning: np.ndarray
) -> np.ndarray:
    z = log_offer - log_b
    onehot = np.eye(len(CATEGORIES))[category]
    return np.column_stack([z, z * has_quote, has_quote, log_offer - log_m, returning, onehot])


@dataclass
class AcceptanceModel:
    lr: LogisticRegression
    kappa: np.ndarray  # typical competitor quote / market price per category
    n_rows: int
    n_test_rows: int

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> AcceptanceModel:
        has = rows.has_quote.to_numpy()
        ratio = np.exp(rows.log_quote - rows.log_m)
        kappa = np.array(
            [
                np.nanmedian(ratio[(rows.category == i) & has]) if ((rows.category == i) & has).any() else 0.12
                for i in range(len(CATEGORIES))
            ]
        )
        log_b, _ = benchmark(
            rows.log_m.to_numpy(), np.exp(rows.log_quote.to_numpy()), rows.quote_age.to_numpy(), rows.category.to_numpy(), kappa
        )
        X = design(
            rows.log_offer.to_numpy(), log_b, has.astype(float), rows.log_m.to_numpy(), rows.category.to_numpy(), rows.returning.to_numpy()
        )
        lr = LogisticRegression(C=10.0, max_iter=2000)
        lr.fit(X, rows.kept.to_numpy())
        return cls(lr, kappa, len(rows), int(rows.test.sum()))

    def prob(
        self, offer: np.ndarray, log_m: np.ndarray, quote: np.ndarray, quote_age: np.ndarray, category: np.ndarray, returning: np.ndarray
    ) -> np.ndarray:
        """P(kept) for each offer (offers broadcast along a trailing grid axis are allowed)."""
        offer = np.asarray(offer, dtype=float)
        log_b, has = benchmark(log_m, quote, quote_age, category, self.kappa)
        if offer.ndim == 2:
            k = offer.shape[1]
            rep = lambda a: np.repeat(np.asarray(a)[:, None], k, axis=1).ravel()  # noqa: E731
            X = design(
                np.log(np.maximum(offer, 1e-3)).ravel(), rep(log_b), rep(has.astype(float)), rep(log_m), rep(category), rep(returning)
            )
            return self.lr.predict_proba(X)[:, 1].reshape(offer.shape)
        X = design(np.log(np.maximum(offer, 1e-3)), log_b, has.astype(float), log_m, category, returning)
        return self.lr.predict_proba(X)[:, 1]

    def benchmark(self, log_m: np.ndarray, quote: np.ndarray, quote_age: np.ndarray, category: np.ndarray) -> np.ndarray:
        return np.exp(benchmark(log_m, quote, quote_age, category, self.kappa)[0])

    def card(self) -> dict:
        c = self.lr.coef_[0]
        return {
            "rows": self.n_rows,
            "test_rows": self.n_test_rows,
            "slope_vs_competitor_quote": round(float(c[0] + c[1]), 3),
            "slope_without_quote": round(float(c[0]), 3),
            "kappa": dict(zip(CATEGORIES, self.kappa.round(3).tolist(), strict=True)),
        }


def acceptance_rows(cat: Catalog, offers: pd.DataFrame, snaps: dict[int, Snapshot], market: MarketPriceModel) -> pd.DataFrame:
    """Offers shown (offer > 0) with what was known about the title on that day."""
    o = offers[offers.offer > 0].copy()
    days = np.array(sorted(snaps))
    o["snap"] = days[np.clip(np.searchsorted(days, o.day.to_numpy(), "right") - 1, 0, len(days) - 1)]
    parts = []
    for d, g in o.groupby("snap"):
        s = snaps[int(d)]
        est = market.estimate(cat, s)
        i = g.idx.to_numpy()
        parts.append(
            pd.DataFrame(
                {
                    "seller_id": g.seller_id.to_numpy(),
                    "basket_id": g.basket_id.to_numpy(),
                    "idx": i,
                    "day": g.day.to_numpy(),
                    "category": cat.category[i],
                    "log_offer": np.log(g.offer.to_numpy()),
                    "log_m": est.log_m[i],
                    "log_quote": np.log(s.quote[i]),
                    "quote_age": (s.quote_age[i] + (g.day.to_numpy() - d)),
                    "has_quote": np.isfinite(s.quote[i]) & ((s.quote_age[i] + (g.day.to_numpy() - d)) <= QUOTE_MAX_AGE),
                    "returning": (g.visit.to_numpy() > 0).astype(float),
                    "kept": g.kept.to_numpy().astype(int),
                    "test": g.test_multiplier.notna().to_numpy(),
                    "offer": g.offer.to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)
