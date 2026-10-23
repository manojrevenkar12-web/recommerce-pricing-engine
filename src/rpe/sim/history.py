"""Generates the 2025 history the company would have: raw exports from its systems (data/raw) and,
separately, the hidden truth (data/truth) for the evaluator.

The current rule-based pricing runs the business, including its randomised price tests. A burn-in
period before 2025 fills the warehouse; it is simulated but only its closing stock is exported, as
the inventory system's opening snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rpe.baseline import Baseline
from rpe.config import RAW, TRUTH, load
from rpe.market import MarketView
from rpe.policy import GRADES
from rpe.sim.run import Simulator
from rpe.sim.world import World, build

HISTORY_SEED = 11


def _dates(w: World, days: pd.Series | np.ndarray) -> pd.Series:
    return pd.to_datetime(w.start) + pd.to_timedelta(np.asarray(days), unit="D")


def generate(raw: Path = RAW, truth: Path = TRUTH) -> dict[str, int]:
    w = build()
    sim = Simulator(w, HISTORY_SEED)
    policy = Baseline(w.new_price, MarketView(w.n), explore=True)
    sim.log.on = False
    burn = sim.run(policy, 0, w.history_start)
    st = burn.state
    opening = pd.DataFrame(
        {
            "unit_id": st.u_id,
            "ean": w.ean[st.u_idx],
            "grade": np.array(GRADES)[st.u_grade],
            "listed_date": _dates(w, st.u_listed).date,
            "price_eur": st.u_price,
            "paid_eur": st.u_cost,
        }
    )
    in_transit = pd.DataFrame(
        {
            "basket_id": st.t_basket,
            "ean": w.ean[st.t_idx],
            "expected_date": _dates(w, st.t_arrive).date,
            "offer_eur": st.t_offer,
        }
    )
    sim.log.on = True
    hist = sim.run(policy, w.history_start, w.history_end - w.history_start + 1, st)
    L = hist.logs

    def ean_of(df: pd.DataFrame) -> pd.Series:
        return pd.Series(w.ean[df["idx"].to_numpy()], index=df.index)

    tables: dict[str, pd.DataFrame] = {}
    tables["catalogue"] = w.catalogue_frame()
    f = L["market_feed"]
    tables["market_feed"] = pd.DataFrame(
        {"date": _dates(w, f.day).date, "ean": ean_of(f), "lowest_used_price": f.price, "n_offers": f.n_offers}
    )
    q = L["buyback_quotes"]
    tables["buyback_quotes"] = pd.DataFrame({"date": _dates(w, q.day).date, "ean": ean_of(q), "competitor_quote": q.quote})
    r = L["sales_ranks"]
    tables["sales_ranks"] = pd.DataFrame(
        {"date": _dates(w, r.day).date, "ean": ean_of(r), "sales_rank": r["rank"].round().astype(np.int64)}
    )
    o = L["offers"].reset_index(drop=True)
    tables["offers"] = pd.DataFrame(
        {
            "offer_id": np.arange(1, len(o) + 1),
            "date": _dates(w, o.day).date,
            "basket_id": o.basket_id,
            "seller_id": o.seller_id,
            "ean": ean_of(o),
            "offer_eur": o.offer.round(2),
            "test_multiplier": o.explore.round(4),
            "kept": o.kept,
        }
    )
    b = L["baskets"]
    tables["baskets"] = pd.DataFrame(
        {
            "basket_id": b.basket_id,
            "date": _dates(w, b.day).date,
            "seller_id": b.seller_id,
            "visit": b.visit,
            "items": b["items"],
            "offered_eur": b.offered_eur,
            "kept_eur": b.kept_eur,
            "sent": b.sent,
        }
    )
    rc = L["receipts"]
    grade_names = np.array([*GRADES, "reject"])
    tables["receipts"] = pd.DataFrame(
        {
            "date": _dates(w, rc.day).date,
            "basket_id": rc.basket_id,
            "ean": ean_of(rc),
            "unit_id": rc.unit_id.where(rc.unit_id > 0).astype("Int64"),
            "grade": grade_names[rc.grade.to_numpy()],
            "paid_eur": rc.paid.round(2),
        }
    )
    p = L["prices"]
    tables["prices"] = pd.DataFrame(
        {"date": _dates(w, p.day).date, "unit_id": p.unit_id, "price_eur": p.price, "test_multiplier": p.explore.round(4)}
    )
    s = L["sales"]
    tables["sales"] = pd.DataFrame({"date": _dates(w, s.day).date, "unit_id": s.unit_id, "price_eur": s.price, "fee_eur": s.fee})
    d = L.get("disposals", pd.DataFrame(columns=["day", "unit_id", "kind"]))
    tables["disposals"] = pd.DataFrame({"date": _dates(w, d.day).date, "unit_id": d.unit_id.astype(np.int64), "kind": d.kind})
    dl = hist.daily
    tables["warehouse_daily"] = pd.DataFrame(
        {"date": _dates(w, dl.day).date, "stock_units": dl.stock, "in_transit_units": dl.in_transit, "capacity_units": w.capacity}
    )
    tables["opening_stock"] = opening
    tables["opening_in_transit"] = in_transit

    injected = _inject_dirty(tables, np.random.default_rng(HISTORY_SEED + 99), load("world")["dirty_data"])

    raw.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(raw / f"{name}.parquet", index=False)
    truth.mkdir(parents=True, exist_ok=True)
    w.truth_frame().to_parquet(truth / "titles.parquet", index=False)
    hist.state.save(truth / "state_end_of_history.pkl")
    meta = {
        "history_seed": HISTORY_SEED,
        "history_ledger": {k: round(v, 2) for k, v in hist.ledger.items()},
        "history_counters": hist.counters,
        "injected_bad_rows": injected,
    }
    (truth / "history.json").write_text(json.dumps(meta, indent=2, default=float))
    return {k: len(v) for k, v in tables.items()}


def _inject_dirty(t: dict[str, pd.DataFrame], rng: np.random.Generator, cfg: dict) -> dict[str, int]:
    """Extra rows that break the contracts, as real exports do. The evaluator checks that ingest
    quarantines every one of them and nothing else."""
    out = {}
    cat = t["catalogue"]
    k = int(len(cat) * cfg["catalogue_bad_ean"])
    bad = cat.sample(k, random_state=1).copy()
    bad["ean"] = [e[:12] + str((int(e[12]) + 1) % 10) for e in bad.ean]
    t["catalogue"] = pd.concat([cat, bad], ignore_index=True)
    out["catalogue"] = k

    feed = t["market_feed"]
    k = int(len(feed) * cfg["feed_bad_price"])
    bad = feed.sample(k, random_state=2).copy()
    bad["lowest_used_price"] = np.where(rng.random(k) < 0.5, -bad.lowest_used_price, bad.lowest_used_price * 10000)
    t["market_feed"] = pd.concat([feed, bad], ignore_index=True)
    out["market_feed"] = k

    off = t["offers"]
    k = int(len(off) * cfg["offers_duplicate"])
    dup = off.sample(k, random_state=3).copy()
    dup["offer_eur"] = (dup.offer_eur + 0.5).round(2)
    t["offers"] = pd.concat([off, dup], ignore_index=True)
    out["offers"] = k

    sales = t["sales"]
    k = max(1, int(len(sales) * cfg["sales_bad_day"]))
    orphan = sales.sample(k, random_state=4).copy()
    orphan["unit_id"] = orphan.unit_id + 10**9
    t["sales"] = pd.concat([sales, orphan], ignore_index=True)
    out["sales"] = k
    return out
