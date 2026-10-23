"""Scores pricing policies against the hidden truth of the simulated market.

This module and rpe.sim are the only readers of data/truth and config/world.yaml. Everything the
engine learns comes from data/raw via the contracts and the warehouse.

A backtest starts every policy from the same end-of-2025 state (stock, parcels, sellers who will
come back) and plays the same sellers and the same market for the forward period. A policy's result
is its cash contribution over the period plus the value of what it holds at the end, both valued
the same way for every policy:

- stock at the end: expected net proceeds if it were sold at the true market price for its
  condition, with true demand, over the rest of its shelf life, else liquidated;
- parcels still on the way: their units valued the same way, less what will be paid for them;
- sellers who will come back: counted, and valued at a fixed amount per returning visit.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpe.baseline import Baseline
from rpe.config import TRUTH, load
from rpe.config import costs as load_costs
from rpe.engine.core import Engine, Flags, hold_value
from rpe.market import MarketView
from rpe.policy import REJECT, Composite, Policy
from rpe.sim.run import Result, SimState, Simulator
from rpe.sim.world import World, build
from rpe.warehouse import table


@dataclass
class Start:
    world: World
    state: SimState
    market: MarketView
    bundle: Any

    @property
    def first_day(self) -> int:
        return self.world.history_end + 1


def load_start(bundle: Any, truth: Path = TRUTH) -> Start:
    w = build()
    cat = bundle.catalog
    if not (cat.ean == w.ean).all():
        raise RuntimeError("the warehouse catalogue does not match the simulated one")
    mv = MarketView.from_history(w.n, table("feed"), table("quotes"), table("ranks"), w.history_end, table("units"))
    return Start(w, SimState.load(truth / "state_end_of_history.pkl"), mv, bundle)


def policy(name: str, s: Start, flags: Flags | None = None) -> Policy:
    def base() -> Baseline:
        return Baseline(s.world.new_price, copy.deepcopy(s.market))

    def eng(f: Flags | None = None, label: str = "engine") -> Engine:
        return Engine(s.bundle, copy.deepcopy(s.market), f or Flags(), name=label)

    if name == "baseline":
        return base()
    if name == "engine":
        return eng(flags)
    if name == "engine_buy_only":
        return Composite(eng(), base(), name)
    if name == "engine_sell_only":
        return Composite(base(), eng(), name)
    if name == "engine_no_market_model":
        return eng(Flags(market_model=False), name)
    if name == "engine_no_capacity":
        return eng(Flags(capacity=False), name)
    if name == "engine_no_long_term":
        return eng(Flags(long_term=False), name)
    raise ValueError(name)


def end_value(w: World, st: SimState, day: int, c: dict[str, Any]) -> dict[str, float]:
    """Truth-based value of stock and parcels at the end of a run (same yardstick for every policy)."""
    lam, m = w.lam(day), w.m(day)
    qf = w.condition_price_factor()
    liq = c["stock"]["liquidation_value"] - c["stock"]["liquidation_handling"]
    hold = c["stock"]["holding_per_unit_day"]
    life = c["stock"]["max_age_days"]
    n_i = np.bincount(st.u_idx, minlength=w.n) + np.bincount(st.t_idx, minlength=w.n)

    def unit_values(idx: np.ndarray, grade: np.ndarray, age: np.ndarray) -> np.ndarray:
        p = np.maximum(m[idx] * qf[grade], c["sell"]["min_price"])
        rel = np.log(p / (m[idx] * qf[grade]))
        mu = lam[idx] * w.share(rel, idx) / np.maximum(n_i[idx], 1)
        v, _, _ = hold_value(p, mu, np.maximum(life - age, 30).astype(float), hold, liq, c)
        return np.maximum(v, liq)

    stock = float(unit_values(st.u_idx, st.u_grade, day - st.u_listed).sum()) if len(st.u_idx) else 0.0
    transit = 0.0
    if len(st.t_idx):
        years = np.maximum(0, w.date_of(day).year - w.pub_year[st.t_idx])
        latent = st.t_latent + w.grade_shift[st.t_idx] + w.cfg["condition"]["age_shift_per_year"] * years + st.s_quality[st.t_seller]
        grade = np.searchsorted(np.array(w.cfg["condition"]["cutpoints"]), latent)
        ok = grade != REJECT
        v = np.zeros(len(grade))
        v[ok] = unit_values(st.t_idx[ok], grade[ok], np.zeros(ok.sum())) - st.t_offer[ok]
        v -= c["buy"]["inbound_handling"]
        v[~ok] -= c["buy"]["reject_disposal"]
        transit = float(v.sum())
    return {"stock_value": stock, "transit_value": transit, "pending_returns": float(len(st.r_seller))}


def score(r: Result, w: World, start_returns: int, cfg: dict[str, Any]) -> dict[str, Any]:
    c = load_costs()
    end = end_value(w, r.state, r.first_day + r.days - 1, c)
    C = r.counters
    value = r.contribution + end["stock_value"] + end["transit_value"]
    per_return = cfg["value_per_pending_return_eur"]
    return {
        "policy": r.policy,
        "seed": r.seed,
        "cash": round(r.contribution, 2),
        "end_stock_value": round(end["stock_value"], 2),
        "end_transit_value": round(end["transit_value"], 2),
        "value": round(value, 2),
        "pending_returns": int(end["pending_returns"]),
        "value_incl_sellers": round(value + per_return * end["pending_returns"], 2),
        "revenue": round(r.ledger["revenue"], 2),
        "revenue_minus_purchases": round(r.ledger["revenue"] + r.ledger["purchase"], 2),
        "units_sold": int(C["units_sold"]),
        "units_bought": int(C["units_received"]),
        "units_liquidated": int(C["units_liquidated"]),
        "avg_sale_price": round(r.ledger["revenue"] / max(C["units_sold"], 1), 2),
        "avg_paid": round(-r.ledger["purchase"] / max(C["units_received"], 1), 2),
        "basket_sent_rate": round(C["baskets_sent"] / max(C["baskets"], 1), 4),
        "item_keep_rate": round(C["items_kept"] / max(C["offers_shown"], 1), 4),
        "marketing_per_new_seller_trade": round(-r.ledger["marketing"] / max(C["new_baskets_sent"], 1), 2),
        "max_fill": round(C["max_fill"], 4),
        "overflow_unit_days": int(C["overflow_unit_days"]),
        "items_bought_while_full": int(C["items_bought_while_full"]),
        "end_stock_units": len(r.state.u_id),
        "ledger": {k: round(v, 2) for k, v in r.ledger.items()},
    }


def run_one(
    s: Start, name: str, seed: int, days: int | None = None, scenario: dict[str, Any] | None = None, audit: bool = False
) -> dict[str, Any]:
    """One policy, one seed, optionally under a scenario: its score, its daily series and, with
    `audit`, the guardrail check of every price it set."""
    cfg = load("release")["backtest"]
    w = s.world.with_scenario(scenario) if scenario else s.world
    days = days or w.cfg["calendar"]["forward_days"]
    t = time.time()
    p = policy(name, s)
    if audit:
        for part in (p, getattr(p, "sell", None)):
            if isinstance(part, Engine):
                part.audit = True
    sim = Simulator(w, seed)
    r = sim.run(p, s.first_day, days, copy.deepcopy(s.state))
    row = score(r, w, len(s.state.r_seller), cfg)
    row["scenario"] = (scenario or {}).get("name", "base")
    row["seconds"] = round(time.time() - t, 1)
    out: dict[str, Any] = {"row": row, "daily": r.daily}
    if audit:
        out["violations"] = audit_prices(p, s)
    return out


def audit_prices(p: Policy, s: Start) -> dict[str, int]:
    from rpe.engine.rules import sell_bounds, sell_violations

    engines = [x for x in (p, getattr(p, "sell", None)) if isinstance(x, Engine)]
    if not engines or not engines[0].audit_log:
        return {}
    e = engines[0]
    a = pd.concat(e.audit_log, ignore_index=True)
    cat = s.bundle.catalog
    b = sell_bounds(e.cfg, cat.category[a.idx.to_numpy()], cat.fmt[a.idx.to_numpy()], a.grade.to_numpy())
    v = sell_violations(e.cfg, e.c, a.price.to_numpy(), a.old.to_numpy(), a.ref.to_numpy(), cat.new_price[a.idx.to_numpy()], b)
    v["prices_checked"] = len(a)
    return v


def run_many(specs: list[dict[str, Any]], workers: int | None = None, start: Start | None = None) -> list[dict[str, Any]]:
    """Runs backtests, in parallel processes when `workers` > 1 (env RPE_WORKERS)."""
    import os

    workers = workers if workers is not None else int(os.environ.get("RPE_WORKERS", "1"))
    if workers <= 1:
        if start is None:
            from rpe.train import Bundle

            start = load_start(Bundle.load())
        return [run_one(start, **sp) for sp in specs]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from rpe.evaluation import worker

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=worker.init) as ex:
        return list(ex.map(worker.run, specs))


def backtest(s: Start, names: list[str], seeds: list[int], days: int | None = None, workers: int | None = None) -> pd.DataFrame:
    res = run_many([{"name": n, "seed": sd, "days": days} for sd in seeds for n in names], workers, s)
    return pd.DataFrame([r["row"] for r in res])


def paired_uplift(df: pd.DataFrame, metric: str, treat: str, control: str = "baseline") -> dict[str, float]:
    """Mean paired difference over seeds with a t-based 95% interval."""
    from scipy import stats

    a = df[df.policy == treat].set_index("seed")[metric]
    b = df[df.policy == control].set_index("seed")[metric]
    d = (a - b).dropna()
    n = len(d)
    mean, base = float(d.mean()), float(b.mean())
    half = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "diff": round(mean, 2),
        "lo": round(mean - half, 2),
        "hi": round(mean + half, 2),
        "control": round(base, 2),
        "pct": round(100 * mean / abs(base), 2) if base else float("nan"),
        "pct_lo": round(100 * (mean - half) / abs(base), 2) if base else float("nan"),
        "pct_hi": round(100 * (mean + half) / abs(base), 2) if base else float("nan"),
        "seeds": n,
    }


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
