"""Does the experimentation framework tell the truth? Checked on the simulated market:

- A/A tests: with no difference between arms, how often does an analysis claim one? By title (the
  unit of randomisation) it should be about 5%; analysing copies as if independent is not.
- Peeking: daily looks with a fixed-horizon test vs the always-valid (mSPRT) p-value.
- A sell-side price test (engine vs rule-based prices, randomised by title): does its 95% interval
  contain the true effect, which the simulator can measure by running both worlds in full?
- CUPED: how much variance does the pre-period covariate remove?
- A buy-side offer test randomised by seller: does the surrogate index (predicted return of the
  seller, available after days) agree with the realised return measured four months later?
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from rpe.baseline import Baseline
from rpe.engine.core import Engine
from rpe.evaluation.backtest import Start
from rpe.experiments.analysis import cuped, srm, welch
from rpe.experiments.assign import arm, seller_arm
from rpe.experiments.sequential import peeking_study
from rpe.experiments.surrogate import realised_effect, surrogate_effect
from rpe.policy import Composite, Split
from rpe.sim.run import Simulator


def _book_value(s: Start) -> np.ndarray:
    """Conservative per-title book value for unsold units at the end of a test: 30% of the last
    competitor price (or of the list-price rule). Observable, identical for both arms."""
    mv = s.market
    fallback = np.where(np.isfinite(s.world.new_price), s.world.new_price * 0.4, 3.99)
    return 0.30 * np.where(np.isfinite(mv.last_price), mv.last_price, fallback)


def cohort_outcomes(s: Start, logs: dict[str, pd.DataFrame], days: int) -> pd.DataFrame:
    """Per unit in stock at the start: what it earned during the test (sale net of fees and
    handling, holding cost, liquidation) plus its book value if still unsold at the end."""
    from rpe.config import costs

    cc = costs()
    st = s.state
    start = s.first_day
    u = pd.DataFrame({"unit_id": st.u_id, "idx": st.u_idx})
    sales = logs.get("sales", pd.DataFrame(columns=["day", "unit_id", "price", "fee"]))
    disp = logs.get("disposals", pd.DataFrame(columns=["day", "unit_id"]))
    u = u.merge(sales[["unit_id", "day", "price", "fee"]].rename(columns={"day": "sold_day"}), on="unit_id", how="left")
    u = u.merge(disp[["unit_id", "day"]].rename(columns={"day": "liq_day"}), on="unit_id", how="left")
    end = np.where(u.sold_day.notna(), u.sold_day, np.where(u.liq_day.notna(), u.liq_day, start + days))
    held = end - start
    y = -cc["stock"]["holding_per_unit_day"] * held
    sold = u.sold_day.notna().to_numpy()
    y = y + np.where(sold, u.price.fillna(0) - u.fee.fillna(0) - cc["sell"]["pick_pack"], 0.0)
    liq = u.liq_day.notna().to_numpy() & ~sold
    y = y + np.where(liq, cc["stock"]["liquidation_value"] - cc["stock"]["liquidation_handling"], 0.0)
    y = y + np.where(~sold & ~liq, _book_value(s)[u.idx.to_numpy()], 0.0)
    u["y"] = y
    return u


def _run(s: Start, policy, seed: int, days: int) -> dict[str, pd.DataFrame]:
    sim = Simulator(s.world, seed, log=True)
    r = sim.run(policy, s.first_day, days, copy.deepcopy(s.state))
    return r.logs


def _pre_period(s: Start) -> np.ndarray:
    """Sales EUR per title in the last 8 weeks of history: a covariate fixed before assignment."""
    from rpe.warehouse import query

    last = s.world.history_end
    q = query(f"select idx, sum(sold_price) as v from units where sold_day > {last - 56} group by 1")
    x = np.zeros(s.world.n)
    x[q.idx.to_numpy()] = q.v.to_numpy()
    return x


def sell_side_test(s: Start, seeds: list[int], days: int = 91, salt0: int = 4711) -> dict[str, Any]:
    """Engine prices vs rule-based prices, randomised by title, on the stock at the start."""
    x_pre = _pre_period(s)
    rows: list[dict[str, Any]] = []
    aa: dict[str, Any] = {}
    for seed in seeds:
        base = lambda: Baseline(s.world.new_price, copy.deepcopy(s.market))  # noqa: E731
        eng = lambda: Engine(s.bundle, copy.deepcopy(s.market))  # noqa: E731
        t_arm = arm(np.arange(s.world.n), salt0 + seed)
        split = Split(base(), eng(), t_arm, name="split")
        y_split = cohort_outcomes(s, _run(s, split, seed, days), days)
        y_c = cohort_outcomes(s, _run(s, base(), seed, days), days)
        y_t = cohort_outcomes(s, _run(s, Composite(base(), eng(), "sell_engine"), seed, days), days)
        per = lambda u: u.groupby("idx").y.sum()  # noqa: E731
        ts, tc, tt = per(y_split), per(y_c), per(y_t)
        titles = ts.index.to_numpy()
        treat = t_arm[titles]
        true_ate = float((tt.reindex(titles) - tc.reindex(titles)).mean())
        est = welch(ts.to_numpy(), treat)
        est_cuped, vr = cuped(ts.to_numpy(), x_pre[titles], treat)
        rows.append(
            {
                "seed": seed,
                "titles": len(titles),
                "srm_p": round(srm(int(treat.sum()), int((~treat).sum())), 4),
                "true_effect_per_title": round(true_ate, 4),
                "estimate": round(est.diff, 4),
                "ci": [round(est.lo, 4), round(est.hi, 4)],
                "covered": bool(est.lo <= true_ate <= est.hi),
                "cuped_estimate": round(est_cuped.diff, 4),
                "cuped_ci": [round(est_cuped.lo, 4), round(est_cuped.hi, 4)],
                "cuped_covered": bool(est_cuped.lo <= true_ate <= est_cuped.hi),
                "cuped_variance_removed": round(vr, 3),
                "relative_effect": round(true_ate / float(tc.reindex(titles).mean()), 4),
            }
        )
        if not aa:
            aa = aa_study(y_c, titles)
    return {
        "per_seed": rows,
        "coverage": round(float(np.mean([r["covered"] for r in rows])), 3),
        "cuped_coverage": round(float(np.mean([r["cuped_covered"] for r in rows])), 3),
        "a_a": aa,
    }


def aa_study(y_units: pd.DataFrame, titles: np.ndarray, n: int = 1000) -> dict[str, float]:
    """False positive rates of A/A splits of one control run: analysed by title vs by copy."""
    by_title = y_units.groupby("idx").y.sum().reindex(titles).to_numpy()
    u_idx, u_y = y_units.idx.to_numpy(), y_units.y.to_numpy()
    fp_title = fp_unit = 0
    for salt in range(n):
        a = arm(titles, 90_000 + salt)
        fp_title += welch(by_title, a).p < 0.05
        au = arm(u_idx, 90_000 + salt)
        fp_unit += welch(u_y, au).p < 0.05
    return {"splits": n, "false_positive_rate_by_title": round(fp_title / n, 4), "false_positive_rate_by_copy": round(fp_unit / n, 4)}


def buy_side_test(s: Start, seed: int = 211, salt: int = 815, test_days: int = 30, follow_days: int = 150) -> dict[str, Any]:
    """Engine offers vs rule-based offers, randomised by seller. Long-term outcome: the seller
    comes back within 120 days of their first basket in the test."""
    from rpe.models.acceptance import benchmark
    from rpe.models.seller_value import HORIZON

    a_of = seller_arm(salt)
    pol = Split(
        Baseline(s.world.new_price, copy.deepcopy(s.market)), Engine(s.bundle, copy.deepcopy(s.market)), None, a_of, name="offer_test"
    )
    logs = _run(s, pol, seed, test_days + follow_days)
    b, o = logs["baskets"], logs["offers"]
    first = b[b.day < s.first_day + test_days].sort_values("day").groupby("seller_id").head(1)
    est = s.bundle.market.estimate_live(s.bundle.catalog, s.market, s.world.history_end)
    oo = o[o.basket_id.isin(first.basket_id) & (o.offer > 0)].copy()
    i = oo.idx.to_numpy()
    lb, _ = benchmark(
        est.log_m[i],
        s.market.quote[i],
        s.world.history_end - s.market.quote_day[i],
        s.bundle.catalog.category[i],
        s.bundle.acceptance.kappa,
    )
    oo["b"] = np.exp(lb)
    sat = oo.groupby("basket_id").agg(o=("offer", "sum"), b=("b", "sum"))
    first = first.merge((sat.o / sat.b).rename("sat"), left_on="basket_id", right_index=True, how="left")
    first["sat"] = first.sat.fillna(0.3)
    later = b[["seller_id", "day"]].rename(columns={"day": "next_day"})
    m = first.merge(later, on="seller_id", how="left")
    m = m[m.next_day > m.day]
    back = m[m.next_day - m.day <= HORIZON].seller_id.unique()
    first["returned"] = first.seller_id.isin(back)
    first["returning"] = first.visit > 0
    first["treat"] = a_of(first.seller_id.to_numpy())
    sur = surrogate_effect(s.bundle.seller, first)
    real = realised_effect(first)
    return {
        "sellers": len(first),
        "srm_p": round(srm(int(first.treat.sum()), int((~first.treat).sum())), 4),
        "surrogate_effect_on_return_rate": {"diff": round(sur.diff, 4), "ci": [round(sur.lo, 4), round(sur.hi, 4)]},
        "realised_effect_on_return_rate": {"diff": round(real.diff, 4), "ci": [round(real.lo, 4), round(real.hi, 4)]},
        "realised_inside_surrogate_ci": bool(sur.lo <= real.diff <= sur.hi),
        "intervals_overlap": bool(max(sur.lo, real.lo) <= min(sur.hi, real.hi)),
        "basket_sent_rate": {
            "treatment": round(float(first[first.treat].sent.mean()), 4),
            "control": round(float(first[~first.treat].sent.mean()), 4),
        },
    }


def evaluate_experiments(s: Start, seeds: list[int]) -> dict[str, Any]:
    return {
        "peeking_null": peeking_study(2000, 28, 100, effect=0.0, seed=1),
        "peeking_with_effect": peeking_study(1000, 28, 100, effect=0.1, seed=2),
        "sell_side_price_test": sell_side_test(s, seeds),
        "buy_side_offer_test": buy_side_test(s),
    }
