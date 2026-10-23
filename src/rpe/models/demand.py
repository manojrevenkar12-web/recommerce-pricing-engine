"""Demand for a listed unit: how fast it sells at a given price. Sellability comes from here.

The daily sale intensity of a unit is

    Z * rate(x) * depth^-d * share(r)      r = log(price / (market price * condition factor))

- rate(x): a gradient-boosted model of the title's demand from what is known about it (category,
  format, age, sales rank, competitor offers, market price, season, our own past sales of it);
- depth^-d: our copies of a title share its buyers (d is estimated, close to 1);
- share(r): a logistic price response per category, the fraction of buyers who pick our listing at
  that relative price;
- Z: what the features cannot know about this copy, a Gamma(k, k) frailty with mean 1. A copy that
  has waited without selling is evidence of a low Z: given exposure H so far, Z ~ Gamma(k, k + H).
  This is what makes the model calibrated over months, and what makes it learn that a copy is a slow
  seller (markdowns and liquidation follow from it).

The price response is estimated only from the units that were in randomised price tests, where the
price was moved independently of demand; the rest of the history would confuse "priced high"
with "a title nobody wants". The parts are fitted in turn (backfitting) until they settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, roots_genlaguerre

from rpe.catalog import CATEGORIES, TITLE_FEATURES, Catalog, title_features
from rpe.models.market_price import MarketPriceModel, Snapshot

UNIT_FEATURES = list(TITLE_FEATURES)
GBM = {
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_data_in_leaf": 400,
    "lambda_l2": 10.0,
    "feature_fraction": 0.9,
    "verbose": -1,
    "seed": 11,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def cloglog_objective(preds: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Gradient and hessian of the negative log-likelihood of "sold within the exposure" when the
    raw score is the log of the expected number of sales: P(sold) = 1 - exp(-exp(score))."""
    y = np.asarray(data.get_label())
    mu = np.exp(np.clip(preds, -30, 8))
    em = np.exp(-mu)
    P = np.maximum(-np.expm1(-mu), 1e-12)
    g1 = mu * em / P  # = -d ll/d score for y = 1
    # d g1 / d score (the hessian of the y = 1 term is its negative)
    dg1 = mu * (em * (1 - mu) * P - mu * em * em) / (P * P)
    grad = np.where(y > 0, -g1, mu)
    hess = np.where(y > 0, -dg1, mu)
    return grad, np.maximum(hess, 1e-6)


@dataclass
class DemandModel:
    booster: lgb.Booster
    beta: np.ndarray  # per category
    alpha: np.ndarray
    log_q: np.ndarray  # condition price factors, very good = 0
    depth_power: float  # our copies of a title share its demand: intensity ~ depth^-depth_power
    frailty_shape: float  # k of the Gamma(k, k) frailty
    history: list[dict] = field(default_factory=list)

    def posterior_scale(self, exposure: np.ndarray) -> np.ndarray:
        """E[Z | not sold after cumulative expected sales `exposure`]."""
        return self.frailty_shape / (self.frailty_shape + np.asarray(exposure, float))

    def quadrature(self, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Nodes x and weights w with E[f(Z)] = sum w f(x / (k + H)) for Z ~ Gamma(k, k + H)."""
        x, wts = roots_genlaguerre(n, self.frailty_shape - 1)
        return x, wts / np.exp(gammaln(self.frailty_shape))

    def log_rate(self, X: pd.DataFrame) -> np.ndarray:
        """log daily sale intensity of a title's demand at share 1 (before splitting it over copies)."""
        return np.asarray(self.booster.predict(X[UNIT_FEATURES], raw_score=True))

    def rel(self, log_price: np.ndarray, log_market: np.ndarray, grade: np.ndarray) -> np.ndarray:
        return log_price - log_market - self.log_q[grade]

    def share(self, r: np.ndarray, category: np.ndarray) -> np.ndarray:
        return _sigmoid(-self.beta[category] * (r - self.alpha[category]))

    def unit_rate(self, X: pd.DataFrame, depth: np.ndarray) -> np.ndarray:
        """Daily intensity of one of `depth` copies at share 1."""
        return np.exp(self.log_rate(X) - self.depth_power * np.log(np.maximum(depth, 1)))

    def intensity(self, X: pd.DataFrame, log_price: np.ndarray, grade: np.ndarray, depth: np.ndarray) -> np.ndarray:
        r = self.rel(log_price, X["log_market"].to_numpy(), grade)
        return self.unit_rate(X, depth) * self.share(r, X["category"].to_numpy())

    def sellability(
        self,
        X: pd.DataFrame,
        grade: np.ndarray,
        depth: np.ndarray,
        days: int = 90,
        log_price: np.ndarray | None = None,
        exposure: np.ndarray | float = 0.0,
    ) -> np.ndarray:
        """P(sold within `days`) at the given price (default: the market price for the condition),
        for a copy that has already waited with cumulative expected sales `exposure`."""
        lp = X["log_market"].to_numpy() + self.log_q[grade] if log_price is None else log_price
        k = self.frailty_shape
        return 1 - (1 + days * self.intensity(X, lp, grade, depth) / (k + np.asarray(exposure, float))) ** (-k)

    # ---------------------------------------------------------------------------------------------
    @classmethod
    def fit(cls, panel: pd.DataFrame, rounds: int = 2, gbm_rounds: int = 300) -> DemandModel:
        panel = panel.sort_values(["unit_id", "week"], kind="stable").reset_index(drop=True)
        X = panel[UNIT_FEATURES]
        y = panel["sold"].to_numpy().astype(float)
        e = panel["exposure"].to_numpy().astype(float)
        lp = panel["log_price"].to_numpy()
        lm = panel["log_market"].to_numpy()
        ld = np.log(np.maximum(panel["depth"].to_numpy(), 1))
        g = panel["grade"].to_numpy().astype(int)
        cat = panel["category"].to_numpy().astype(int)
        test = panel["test"].to_numpy().astype(bool)
        unit = panel["unit_id"].to_numpy()
        last = np.r_[unit[1:] != unit[:-1], True]
        beta, alpha, log_q, theta, k = np.full(len(CATEGORIES), 4.0), np.zeros(len(CATEGORIES)), np.zeros(4), 1.0, 2.0
        H_before = np.zeros(len(panel))
        hist = []
        booster = None
        for it in range(rounds):
            lz = np.log(k / (k + H_before))
            r = lp - lm - log_q[g]
            log_s = -np.logaddexp(0, beta[cat] * (r - alpha[cat]))
            ds = lgb.Dataset(
                X, y, init_score=np.log(e) + log_s - theta * ld + lz, categorical_feature=["category", "fmt"], free_raw_data=False
            )
            booster = lgb.train({**GBM, "objective": cloglog_objective}, ds, gbm_rounds)
            f = np.asarray(booster.predict(X, raw_score=True))
            beta, alpha, log_q, theta, nll = _fit_response(
                y[test], e[test], f[test] + lz[test], lp[test] - lm[test], ld[test], g[test], cat[test], beta, alpha, log_q, theta
            )
            r = lp - lm - log_q[g]
            h = e * np.exp(f - np.logaddexp(0, beta[cat] * (r - alpha[cat])) - theta * ld)
            H_before = pd.Series(h).groupby(unit).cumsum().to_numpy() - h
            if it == rounds - 1:
                # the frailty must measure what the features cannot predict out of sample: refit the
                # rate model on half the titles, predict the other half, and estimate k on that
                fold = panel["idx"].to_numpy() % 2 == 0
                f_oof = np.zeros(len(panel))
                for part in (fold, ~fold):
                    sub = lgb.Dataset(
                        X[part], y[part], init_score=(np.log(e) + log_s - theta * ld + lz)[part], categorical_feature=["category", "fmt"]
                    )
                    b_ = lgb.train({**GBM, "objective": cloglog_objective}, sub, gbm_rounds)
                    f_oof[~part] = np.asarray(b_.predict(X[~part], raw_score=True))
                h_oof = e * np.exp(f_oof - np.logaddexp(0, beta[cat] * (r - alpha[cat])) - theta * ld)
                Hb_oof = pd.Series(h_oof).groupby(unit).cumsum().to_numpy() - h_oof
                k = _fit_frailty(Hb_oof[last], h_oof[last], y[last])
            else:
                k = _fit_frailty(H_before[last], h[last], y[last])
            hist.append(
                {
                    "round": it,
                    "nll_test_rows": round(float(nll), 2),
                    "beta": beta.round(3).tolist(),
                    "depth_power": round(theta, 3),
                    "frailty_shape": round(k, 3),
                }
            )
        assert booster is not None
        return cls(booster, beta, alpha, log_q, theta, k, hist)


def _fit_frailty(H: np.ndarray, h: np.ndarray, y: np.ndarray) -> float:
    """Shape k of a Gamma(k, k) frailty by maximum likelihood on each unit's history: cumulative
    expected sales H before its last week, h in it, and whether it sold then."""

    def surv(x: np.ndarray, k: float) -> np.ndarray:
        return np.exp(-k * np.log1p(x / k))

    def nll(logk: float) -> float:
        k = float(np.exp(logk))
        s0, s1 = surv(H, k), surv(H + h, k)
        ll = np.where(y > 0, np.log(np.maximum(s0 - s1, 1e-300)), np.log(np.maximum(s1, 1e-300)))
        return float(-ll.sum())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(200.0)), method="bounded")
    return float(np.exp(res.x))


def _fit_response(y, e, f, r0, ld, g, cat, beta0, alpha0, logq0, theta0):
    """Maximum likelihood for the price response (beta, alpha per category), the condition factors
    and how copies share demand, given the rate model, on rows where the price was randomised. Weak
    priors keep thin categories sensible: beta ~ N(4, 2^2), alpha ~ N(0, 0.2^2), depth power ~ N(1, 0.5^2)."""
    K = len(beta0)

    def unpack(th):
        return np.exp(th[:K]), th[K : 2 * K], np.concatenate([th[2 * K : 2 * K + 1], [0.0], th[2 * K + 1 : 2 * K + 3]]), th[2 * K + 3]

    def nll(th):
        beta, alpha, lq, theta = unpack(th)
        r = r0 - lq[g]
        z = beta[cat] * (r - alpha[cat])
        eta = np.log(e) + f - np.logaddexp(0, z) - theta * ld
        mu = np.exp(eta)
        P = -np.expm1(-mu)
        ll = np.where(y > 0, np.log(np.maximum(P, 1e-300)), -mu)
        dll = np.where(y > 0, mu * np.exp(-mu) / np.maximum(P, 1e-300), -mu)  # d ll / d eta
        s = _sigmoid(z)  # d softplus(z)/dz
        # d eta / d log beta = -s * z ; d eta / d alpha = s * beta ; d eta / d logq_g = s * beta ; d eta / d theta = -ld
        gb = np.bincount(cat, weights=dll * (-s * z), minlength=K)
        ga = np.bincount(cat, weights=dll * (s * beta[cat]), minlength=K)
        gq_all = np.bincount(g, weights=dll * (s * beta[cat]), minlength=4)
        gt = (dll * -ld).sum()
        prior = ((beta - 4) ** 2).sum() / (2 * 4) + (alpha**2).sum() / (2 * 0.04) + (theta - 1) ** 2 / (2 * 0.25)
        grad = np.concatenate([-gb + (beta - 4) / 4 * beta, -ga + alpha / 0.04, -gq_all[[0]], -gq_all[2:], [-gt + (theta - 1) / 0.25]])
        return -ll.sum() + prior, grad

    th0 = np.concatenate([np.log(beta0), alpha0, logq0[[0]], logq0[2:], [theta0]])
    res = minimize(nll, th0, jac=True, method="L-BFGS-B")
    beta, alpha, lq, theta = unpack(res.x)
    return beta, alpha, lq, float(theta), res.fun


# -------------------------------------------------------------------------------------------------
def build_panel(
    cat: Catalog, units: pd.DataFrame, prices: pd.DataFrame, snaps: dict[int, Snapshot], market: MarketPriceModel, end_day: int
) -> pd.DataFrame:
    """Unit-weeks: one row per unit and week it was listed (weeks start on the snapshot days)."""
    days = np.array(sorted(snaps))
    u = units.copy()
    u["end"] = np.select(
        [u.sold_day.notna(), u.disposed_day.notna()],
        [u.sold_day.fillna(0) + 1, u.disposed_day.fillna(0)],
        end_day + 1,
    ).astype(int)
    u = u[u.end > days[0]]
    first = np.searchsorted(days, np.maximum(u.listed_day.to_numpy(), days[0]), "right") - 1
    first = np.maximum(first, 0)
    last = np.searchsorted(days, u.end.to_numpy() - 1, "right") - 1
    n_w = np.maximum(last - first + 1, 0)
    rep = np.repeat(np.arange(len(u)), n_w)
    wk = np.concatenate([np.arange(a, a + k) for a, k in zip(first, n_w, strict=True)]) if len(u) else np.zeros(0, int)
    P = pd.DataFrame(
        {
            "unit_id": u.unit_id.to_numpy()[rep],
            "idx": u.idx.to_numpy()[rep],
            "grade": u.grade_code.to_numpy()[rep],
            "listed": u.listed_day.to_numpy()[rep],
            "end": u.end.to_numpy()[rep],
            "sold_day": u.sold_day.to_numpy()[rep],
            "week": days[wk],
        }
    )
    P["start"] = np.maximum(P.week, P.listed)
    P["stop"] = np.minimum(P.week + 7, P.end)
    P = P[P.stop > P.start].copy()
    P["exposure"] = P.stop - P.start
    P["sold"] = (P.sold_day.notna() & (P.sold_day >= P.start) & (P.sold_day < P.stop)).astype(int)
    # price in force at the start of the interval
    pr = prices[["day", "unit_id", "price", "test_multiplier"]].astype(
        {"day": "int64", "unit_id": "int64", "price": float, "test_multiplier": float}
    )
    pr = pr.sort_values("day")
    P = P.astype({"start": "int64", "unit_id": "int64"}).sort_values("start")
    P = pd.merge_asof(P, pr, left_on="start", right_on="day", by="unit_id", direction="backward")
    P = P[P.price.notna()].copy()
    P["test"] = P.test_multiplier.notna()
    # copies of the same title listed at the start of the week
    depth = {}
    for d in days:
        live = (u.listed_day <= d) & (u.end > d)
        depth[d] = np.bincount(u.idx[live], minlength=cat.n)
    out = []
    for d, g in P.groupby("week"):
        s = snaps[int(d)]
        est = market.estimate(cat, s)
        X = title_features(cat, s, int(d), est.log_m, s.n, g.idx.to_numpy())
        X.index = g.index
        X["days_in_stock"] = (g.start - g.listed).to_numpy()
        X["depth"] = np.maximum(depth[int(d)][g.idx.to_numpy()], 1)
        out.append(pd.concat([g[["unit_id", "idx", "grade", "week", "exposure", "sold", "price", "test"]], X], axis=1))
    panel = pd.concat(out).reset_index(drop=True)
    panel["log_price"] = np.log(panel.price)
    return panel
