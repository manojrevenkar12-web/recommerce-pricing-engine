"""Long-term seller value: does a better offer bring the seller back?

Basket-level logistic model of "the seller comes back with another basket within 120 days" on how
the offers compared with the competitor (satisfaction), whether the basket was sent, and its value.
Its slope on satisfaction, times what a returning seller's basket is worth, is the long-term value
of a more generous offer: the engine adds it to the buy decision, and experiments use the model as a
surrogate for the long-term outcome they cannot wait for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rpe.config import costs

HORIZON = 120


def basket_features(sat: np.ndarray, sent: np.ndarray, kept_eur: np.ndarray, returning: np.ndarray) -> np.ndarray:
    ls = np.log(np.clip(sat, 0.05, 20))
    return np.column_stack([ls, sent, ls * sent, np.log1p(kept_eur / 10) * sent, returning])


@dataclass
class SellerValueModel:
    lr: LogisticRegression
    value_per_return: float  # contribution a returning seller's visit brings, EUR
    base_rate: float
    rows: int

    @classmethod
    def fit(
        cls, baskets: pd.DataFrame, offers: pd.DataFrame, acc_rows: pd.DataFrame, units: pd.DataFrame, last_day: int
    ) -> SellerValueModel:
        b = baskets.sort_values(["seller_id", "day"]).copy()
        b["next_day"] = b.groupby("seller_id").day.shift(-1)
        b["returned"] = (b.next_day - b.day <= HORIZON).fillna(False).astype(int)
        sat = satisfaction(acc_rows)
        b = b.merge(sat, on="basket_id", how="left")
        b["sat"] = b.sat.fillna(0.3)
        obs = b[b.day <= last_day - HORIZON]
        X = basket_features(
            obs.sat.to_numpy(), obs.sent.to_numpy().astype(float), obs.kept_eur.to_numpy(), (obs.visit > 0).to_numpy().astype(float)
        )
        lr = LogisticRegression(max_iter=2000, C=10.0)
        lr.fit(X, obs.returned.to_numpy())
        return cls(lr, _value_per_visit(baskets, units, last_day), float(obs.returned.mean()), len(obs))

    def p_return(self, sat: np.ndarray, sent: np.ndarray, kept_eur: np.ndarray, returning: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(basket_features(sat, sent, kept_eur, returning))[:, 1]

    def marginal_value(self, sat: float = 1.0, kept_eur: float = 15.0) -> float:
        """EUR of future contribution per unit increase of log satisfaction for a sent basket."""
        c = self.lr.coef_[0]
        p = float(self.p_return(np.array([sat]), np.array([1.0]), np.array([kept_eur]), np.array([0.0]))[0])
        return float((c[0] + c[2]) * p * (1 - p) * self.value_per_return)

    def card(self) -> dict:
        c = self.lr.coef_[0]
        return {
            "rows": self.rows,
            "return_rate_120d": round(self.base_rate, 3),
            "slope_log_satisfaction_sent": round(float(c[0] + c[2]), 3),
            "value_per_return_eur": round(self.value_per_return, 2),
            "marginal_value_eur_per_log_sat": round(self.marginal_value(), 2),
        }


def satisfaction(acc_rows: pd.DataFrame) -> pd.DataFrame:
    """Offered total / competitor benchmark total per basket, over items we made an offer for."""
    from rpe.models.acceptance import benchmark

    kappa = acc_rows.attrs.get("kappa")
    if kappa is None:
        has = acc_rows.has_quote.to_numpy()
        kappa = np.array([np.nanmedian(np.exp(acc_rows.log_quote - acc_rows.log_m)[(acc_rows.category == i) & has]) for i in range(10)])
    log_b, _ = benchmark(
        acc_rows.log_m.to_numpy(), np.exp(acc_rows.log_quote.to_numpy()), acc_rows.quote_age.to_numpy(), acc_rows.category.to_numpy(), kappa
    )
    g = pd.DataFrame({"basket_id": acc_rows.basket_id, "o": acc_rows.offer, "b": np.exp(log_b)}).groupby("basket_id").sum()
    return pd.DataFrame({"basket_id": g.index, "sat": (g.o / g.b).to_numpy()})


def _value_per_visit(baskets: pd.DataFrame, units: pd.DataFrame, last_day: int) -> float:
    """Average realised contribution of a basket by a returning seller (sold units net of fees and
    handling, minus what was paid, the parcel and the inbound handling), counted on baskets old enough
    for most units to have sold; unsold units are valued at liquidation."""
    c = costs()
    early = baskets[(baskets.day <= last_day - 180)]
    ret = early[early.visit > 0]
    if len(ret) == 0:
        return 0.0
    from rpe.warehouse import query

    r = query(
        "select r.basket_id, count(*) as received, sum(coalesce(u.sold_price - u.fee_eur, 0)) as net_sales, "
        "count(u.sold_day) as sold, sum(r.paid_eur) as paid from receipts r left join units u using (unit_id) group by 1"
    )
    m = ret.merge(r, on="basket_id", how="left").fillna(0)
    unsold = m.received - m.sold
    value = (
        m.net_sales
        - c["sell"]["pick_pack"] * m.sold
        - m.paid
        - c["buy"]["inbound_handling"] * m.received
        - np.where(m.sent, c["buy"]["basket_shipping"], 0.0)
        + (c["stock"]["liquidation_value"] - c["stock"]["liquidation_handling"]) * unsold
    )
    return float(value.mean())
