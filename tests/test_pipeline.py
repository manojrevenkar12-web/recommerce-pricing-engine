"""History, contracts, warehouse and models (needs the generated data)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from rpe.config import CLEAN, TRUTH
from rpe.warehouse import query

pytestmark = pytest.mark.pipeline
SRC = Path(__file__).resolve().parents[1] / "src" / "rpe"


def test_only_simulator_and_evaluation_read_the_truth():
    """Models, engine, API and steering never see the simulated truth. (The command line only
    chains the steps; the simulator and the evaluation are its separate stages.)"""
    allowed = {"sim", "evaluation", "cli.py"}
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC)
        if rel.parts[0] in allowed or rel.name == "config.py":
            continue
        text = p.read_text()
        if re.search(r"\bTRUTH\b|data/truth|load\(\"world\"\)|world\.yaml|rpe\.sim|from rpe\.evaluation", text):
            offenders.append(str(rel))
    assert offenders == [], offenders


def test_every_injected_bad_row_is_quarantined(built):
    injected = json.loads((TRUTH / "history.json").read_text())["injected_bad_rows"]
    rep = {s["source"]: s for s in json.loads((CLEAN / "_ingest_report.json").read_text())["sources"]}
    for source, n in injected.items():
        assert rep[source]["quarantined"] == n, source
    assert sum(s["quarantined"] for s in rep.values()) == sum(injected.values())


def test_units_are_accounted_for(built):
    q = query(
        "select count(*) as n, count(sold_day) as sold, count(disposed_day) as liq, count(*) filter (where sold_day is not null and disposed_day is not null) as both_ from units"
    )
    assert q.both_[0] == 0
    last = query("select * from warehouse_daily order by day desc limit 1")
    in_stock = q.n[0] - q.sold[0] - q.liq[0]
    assert in_stock == last.stock_units[0]


def test_randomised_tests_are_in_the_history(built):
    q = query("select count(*) filter (where test_multiplier is not null) * 1.0 / count(*) as share from offers where offer > 0")
    assert 0.10 < q.share[0] < 0.20


def test_model_card(bundle):
    c = bundle.card
    assert c["rows"]["demand_test_rows"] > 50_000
    assert 0.5 < c["condition_factors"]["acceptable"] < c["condition_factors"]["good"] < 1.0 < c["condition_factors"]["like_new"]
    assert all(1.0 < v["beta"] < 10 for v in c["price_response"].values())
    assert 0.6 < bundle.demand.depth_power < 1.4


def test_demand_responds_to_price_and_copies(bundle, start):
    from rpe.engine.core import Engine

    e = Engine(bundle, start.market)
    day = start.first_day
    idx = np.arange(0, bundle.catalog.n, 97)
    est = e.estimate(day)
    X = e.features(day, idx, np.zeros(len(idx)), est)
    g = np.ones(len(idx), int)
    lp = est.log_m[idx]
    low = bundle.demand.intensity(X, lp - 0.2, g, np.ones(len(idx)))
    high = bundle.demand.intensity(X, lp + 0.2, g, np.ones(len(idx)))
    many = bundle.demand.intensity(X, lp, g, np.full(len(idx), 10))
    one = bundle.demand.intensity(X, lp, g, np.ones(len(idx)))
    assert np.all(low > high) and np.all(many < one)
    s = bundle.demand.sellability(X, g, np.ones(len(idx)))
    assert np.all((s >= 0) & (s <= 1))


def test_market_price_beats_last_competitor_price(start):
    from rpe.evaluation.models import market_price

    r = market_price(start)
    assert r["feed_titles"]["engine"]["mean_abs_log_error"] < r["feed_titles"]["rule_last_price"]["mean_abs_log_error"]
    assert r["cold_start_titles"]["engine"]["mean_abs_log_error"] < r["cold_start_titles"]["rule_share_of_list_price"]["mean_abs_log_error"]


def test_acceptance_rises_with_offer(bundle, start):
    i = np.arange(0, bundle.catalog.n, 211)
    est = bundle.market.estimate_live(bundle.catalog, start.market, start.world.history_end)
    G = np.outer(np.ones(len(i)), [0.2, 0.5, 1.0, 2.0])
    p = bundle.acceptance.prob(G, est.log_m[i], start.market.quote[i], np.zeros(len(i)), bundle.catalog.category[i], np.zeros(len(i)))
    assert np.all(np.diff(p, axis=1) > 0)
