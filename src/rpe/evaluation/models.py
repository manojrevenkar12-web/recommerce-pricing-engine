"""Model quality against the truth: market price, demand and price response, sellability, seller
acceptance and grading. Each model is compared with what the rule-based pricing assumes today."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from rpe.baseline import Baseline
from rpe.catalog import CATEGORIES
from rpe.engine.core import Engine
from rpe.evaluation.backtest import Start
from rpe.policy import Items, MarketUpdate, OfferDecision, PriceDecision, Units, Warehouse
from rpe.sim.run import Simulator


def _err(est: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    ape = np.abs(est / truth - 1)
    return {
        "median_abs_pct_error": round(float(np.median(ape)) * 100, 2),
        "mean_abs_log_error": round(float(np.mean(np.abs(np.log(est / truth)))), 4),
        "titles": len(est),
    }


def market_price(s: Start) -> dict[str, Any]:
    w, b = s.world, s.bundle
    day = w.history_end
    est = b.market.estimate_live(b.catalog, s.market, day)
    truth = w.m(day)
    released = w.release_day <= day
    feed = released & s.market.ever_seen
    cold = released & ~s.market.ever_seen
    raw = s.market.last_price
    rule = np.where(np.isfinite(b.catalog.new_price), b.catalog.new_price * 0.4, 3.99)
    lo, hi = np.exp(est.log_p10), np.exp(est.log_p90)
    return {
        "feed_titles": {"engine": _err(est.m[feed], truth[feed]), "rule_last_price": _err(raw[feed], truth[feed])},
        "cold_start_titles": {
            "engine": _err(est.m[cold], truth[cold]),
            "rule_share_of_list_price": _err(rule[cold], truth[cold]),
            "p10_p90_coverage": round(float(((truth[cold] >= lo[cold]) & (truth[cold] <= hi[cold])).mean()), 3),
        },
    }


def price_response(s: Start) -> dict[str, Any]:
    w, d = s.world, s.bundle.demand
    names = w.categories
    true_beta = {c: w.cfg["catalogue"]["categories"][c]["beta"] for c in names}
    out = {c: {"estimated": round(float(d.beta[CATEGORIES.index(c)]), 3), "true": true_beta[c]} for c in CATEGORIES}
    rel = np.array([v["estimated"] / v["true"] - 1 for v in out.values()])
    qtrue = w.cfg["condition"]["price_factor"]
    return {
        "beta": out,
        "beta_max_abs_rel_error": round(float(np.abs(rel).max()), 3),
        "beta_mean_abs_rel_error": round(float(np.abs(rel).mean()), 3),
        "depth_power": {"estimated": round(d.depth_power, 3), "true": 1.0},
        "condition_factors": {
            g: {"estimated": round(float(np.exp(d.log_q[i])), 3), "true": qtrue[g]}
            for i, g in enumerate(["like_new", "very_good", "good", "acceptable"])
        },
    }


def demand_calibration(s: Start) -> dict[str, Any]:
    """Model vs true sale intensity of every unit in stock at the start of the forward period."""
    w, b = s.world, s.bundle
    e = Engine(b, copy.deepcopy(s.market))
    day = s.first_day
    st = s.state
    idx, g, age = st.u_idx, st.u_grade, day - st.u_listed
    depth = np.bincount(idx, minlength=w.n)[idx]
    est = e.estimate(day)
    X = e.features(day, idx, age, est)
    model = b.demand.intensity(X, np.log(st.u_price), g, depth)
    rel = np.log(st.u_price / (w.m(day)[idx] * w.condition_price_factor()[g]))
    true = w.lam(day)[idx] * w.share(rel, idx) / depth
    df = pd.DataFrame(
        {
            "m": model,
            "t": true,
            "age": pd.cut(age, [-1, 30, 90, 180, 10_000], labels=["0-30", "31-90", "91-180", ">180"]),
            "cat": np.array(CATEGORIES)[b.catalog.category[idx]],
        }
    )
    ratio = lambda g_: {str(k): round(float(v.m.sum() / v.t.sum()), 3) for k, v in df.groupby(g_, observed=True)}  # noqa: E731
    return {
        "overall_ratio": round(float(model.sum() / true.sum()), 3),
        "by_age_days": ratio("age"),
        "by_category": ratio("cat"),
        "units": len(idx),
    }


class ReferencePricer:
    """Lists every unit at the engine's market price for its condition and buys like the rule-based
    policy: a fixed price over 90 days lets predicted sellability be checked against sales."""

    name = "reference"

    def __init__(self, s: Start):
        self.e = Engine(s.bundle, copy.deepcopy(s.market))
        self.b = Baseline(s.world.new_price, copy.deepcopy(s.market))
        self.min_price = self.e.c["sell"]["min_price"]

    def observe(self, u: MarketUpdate) -> None:
        self.e.observe(u)
        self.b.observe(u)

    def price(self, day: int, units: Units, wh: Warehouse) -> PriceDecision:
        p = units.price.copy()
        new = np.isnan(p)
        if new.any():
            est = self.e.estimate(day)
            p[new] = np.maximum(np.round(est.m[units.idx[new]] * self.e.q[units.grade[new]], 2), self.min_price)
        return PriceDecision(p, np.zeros(len(units), bool))

    def offer(self, day: int, items: Items, wh: Warehouse) -> OfferDecision:
        return self.b.offer(day, items, wh)


def sellability_and_grading(s: Start, seed: int = 301, days: int = 90) -> dict[str, Any]:
    w, b = s.world, s.bundle
    st0 = copy.deepcopy(s.state)
    pol = ReferencePricer(s)
    day = s.first_day
    # every unit in stock is re-listed at the reference price on day one
    est = pol.e.estimate(day)
    st0.u_price = np.maximum(np.round(est.m[st0.u_idx] * pol.e.q[st0.u_grade], 2), pol.min_price)
    idx, g, age = st0.u_idx, st0.u_grade, day - st0.u_listed
    depth = np.bincount(idx, minlength=w.n)[idx]
    X = pol.e.features(day, idx, age, est)
    # exposure so far, as the engine reckons it: expected sales at today's price over the days listed
    r_now = np.log(st0.u_price) - est.log_m[idx] - b.demand.log_q[g]
    exposure = b.demand.unit_rate(X, depth) * b.demand.share(r_now, b.catalog.category[idx]) * age
    pred = b.demand.sellability(X, g, depth, days, np.log(st0.u_price), exposure)
    ids0 = st0.u_id.copy()
    sim = Simulator(w, seed, log=True)
    r = sim.run(pol, day, days, st0)
    sold_ids = set(r.logs["sales"].unit_id.tolist()) if "sales" in r.logs else set()
    y = np.isin(ids0, list(sold_ids)).astype(float)
    bins = np.clip((pred * 10).astype(int), 0, 9)
    ece = sum(abs(pred[bins == k].mean() - y[bins == k].mean()) * (bins == k).mean() for k in range(10) if (bins == k).any())
    auc = _auc(pred, y)
    rule_score = -np.nan_to_num(s.market.rank[idx], nan=1e9)  # what the rules use: sales rank
    rec = r.logs["receipts"]
    gp = b.grading.probs(b.catalog, rec.idx.to_numpy(), day)
    yy = rec.grade.to_numpy()
    ll_model = float(-np.mean(np.log(np.maximum(gp[np.arange(len(yy)), yy], 1e-9))))
    freq = np.bincount(yy, minlength=5) / len(yy)
    ll_freq = float(-np.mean(np.log(freq[yy])))
    return {
        "sellability_90d": {
            "units": len(y),
            "predicted_mean": round(float(pred.mean()), 4),
            "realised_mean": round(float(y.mean()), 4),
            "expected_calibration_error": round(float(ece), 4),
            "auc": round(auc, 4),
            "auc_sales_rank_rule": round(_auc(rule_score, y), 4),
        },
        "grading": {"receipts": len(yy), "log_loss_model": round(ll_model, 4), "log_loss_overall_rates": round(ll_freq, 4)},
    }


def acceptance(s: Start, sample: int = 50_000) -> dict[str, Any]:
    """Predicted P(seller keeps the item) against the true probability, for the offers of the last
    four weeks of history (valued with the market view at the end of history)."""
    from rpe.warehouse import query

    w, b = s.world, s.bundle
    last = w.history_end
    o = query(
        f"select o.day, o.idx, o.offer, b.visit from offers o join baskets b using (basket_id) where o.offer > 0 and o.day > {last - 28}"
    )
    o = o.sample(min(sample, len(o)), random_state=7)
    sc = w.cfg["sellers"]
    sigma = np.sqrt(sc["reservation_sigma"] ** 2 + sc["seller_factor_sigma"] ** 2)
    i = o.idx.to_numpy()
    true = stats.norm.cdf((np.log(o.offer.to_numpy()) - np.log(w.quote(last)[i])) / sigma)
    est = b.market.estimate_live(b.catalog, s.market, last)
    qage = last - s.market.quote_day[i]
    p = b.acceptance.prob(
        o.offer.to_numpy(), est.log_m[i], s.market.quote[i], qage, b.catalog.category[i], (o.visit.to_numpy() > 0).astype(float)
    )
    df = pd.DataFrame({"t": true, "p": p})
    return {
        "offers": len(df),
        "mean_abs_error": round(float(np.abs(df.p - df.t).mean()), 4),
        "predicted_mean": round(float(df.p.mean()), 4),
        "true_mean": round(float(df.t.mean()), 4),
    }


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    r = stats.rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def evaluate_models(s: Start) -> dict[str, Any]:
    return {
        "market_price": market_price(s),
        "price_response": price_response(s),
        "demand_calibration": demand_calibration(s),
        **sellability_and_grading(s),
        "acceptance": acceptance(s),
    }
