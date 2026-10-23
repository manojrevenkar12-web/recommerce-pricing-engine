"""Engine decisions and the simulator (needs the generated data)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rpe.baseline import Baseline
from rpe.engine.core import Engine, Flags
from rpe.policy import Items, Units, Warehouse
from rpe.sim.run import Simulator

pytestmark = pytest.mark.pipeline


def _items(idx, visit=0):
    n = len(idx)
    return Items(np.arange(n) // 20, np.arange(n) // 20, np.full(n, visit), np.asarray(idx))


def _wh(start, stock=20_000, transit=500, depth=None):
    return Warehouse(stock, transit, start.world.capacity, depth if depth is not None else np.zeros(start.world.n, int))


def test_offers_respect_caps_and_reasons(bundle, start):
    e = Engine(bundle, copy.deepcopy(start.market))
    day = start.first_day
    idx = np.arange(0, bundle.catalog.n, 13)
    d = e.offer_details(day, _items(idx), _wh(start))
    est = e.estimate(day)
    cap = e.cfg["buy"]["max_offer_share"] * np.exp(est.log_m[idx])
    assert np.all(d.offer.to_numpy() <= cap + 0.011)
    assert set(d.reason) <= {"buy", "no_margin", "slow_seller", "warehouse_full"}
    assert np.all((d.offer > 0) == (d.reason == "buy"))
    assert np.all(d.offer[d.offer > 0] >= e.c["buy"]["min_offer"])
    assert (d.reason == "buy").mean() > 0.1


def test_more_copies_lower_the_offer(bundle, start):
    # without the long-term term, which couples the items of a basket
    e = Engine(bundle, copy.deepcopy(start.market), Flags(long_term=False))
    day = start.first_day
    idx = np.arange(0, bundle.catalog.n, 17)
    none = e.offer_details(day, _items(idx), _wh(start)).offer.to_numpy()
    depth = np.zeros(start.world.n, int)
    depth[idx] = 25
    many = e.offer_details(day, _items(idx), _wh(start, depth=depth)).offer.to_numpy()
    assert np.all(many <= none + 1e-9) and many.sum() < none.sum()


def test_long_term_value_raises_offers(bundle, start):
    idx = np.arange(0, bundle.catalog.n, 17)
    with_lt = Engine(bundle, copy.deepcopy(start.market)).offer_details(start.first_day, _items(idx), _wh(start)).offer.to_numpy()
    without = (
        Engine(bundle, copy.deepcopy(start.market), Flags(long_term=False))
        .offer_details(start.first_day, _items(idx), _wh(start))
        .offer.to_numpy()
    )
    assert np.all(with_lt[without > 0] >= without[without > 0] - 1e-9) and with_lt.sum() > without.sum()


def test_full_warehouse_stops_buying(bundle, start):
    e = Engine(bundle, copy.deepcopy(start.market))
    cap = start.world.capacity
    d = e.offer_details(start.first_day, _items(np.arange(0, 2000, 7)), Warehouse(cap, 0, cap, np.zeros(start.world.n, int)))
    assert (d.offer == 0).all() and (d.reason == "warehouse_full").all()


def test_sell_prices_within_guardrails_and_young_stock_kept(bundle, start):
    from rpe.engine.rules import sell_bounds, sell_violations

    e = Engine(bundle, copy.deepcopy(start.market), audit=True)
    st = start.state
    n = len(st.u_id)
    units = Units(st.u_id, st.u_idx, st.u_grade, st.u_listed, st.u_price, np.ones(n, bool))
    dec = e.price(start.first_day, units, _wh(start, depth=np.bincount(st.u_idx, minlength=start.world.n)))
    a = e.audit_log[0]
    b = sell_bounds(e.cfg, bundle.catalog.category[a.idx.to_numpy()], bundle.catalog.fmt[a.idx.to_numpy()], a.grade.to_numpy())
    v = sell_violations(e.cfg, e.c, a.price.to_numpy(), a.old.to_numpy(), a.ref.to_numpy(), bundle.catalog.new_price[a.idx.to_numpy()], b)
    assert sum(v.values()) == 0, v
    age = start.first_day - st.u_listed
    assert not dec.liquidate[age < e.cfg["sell"]["liquidate_min_age_days"]].any()
    assert dec.liquidate.sum() <= e.cfg["sell"]["max_liquidation_share_per_week"] * n + (age >= e.cfg["sell"]["max_age_days"]).sum()


def test_simulator_common_random_numbers_and_books(bundle, start):
    """Two policies meet the same sellers with the same items; the ledger adds up."""
    seen = {}

    class Spy(Baseline):
        def offer(self, day, items, wh):
            seen.setdefault(self.name, []).append(items.idx.copy())
            return super().offer(day, items, wh)

    a = Spy(start.world.new_price, copy.deepcopy(start.market))
    a.name = "a"
    b = Spy(start.world.new_price, copy.deepcopy(start.market))
    b.name = "b"
    b.cfg = {**b.cfg, "buy": {**b.cfg["buy"], "rates": [r * 1.5 for r in b.cfg["buy"]["rates"]]}}
    ra = Simulator(start.world, 7).run(a, start.first_day, 3, copy.deepcopy(start.state))
    rb = Simulator(start.world, 7).run(b, start.first_day, 3, copy.deepcopy(start.state))
    assert np.array_equal(seen["a"][0], seen["b"][0])  # day one: identical sellers and items
    assert ra.contribution == pytest.approx(sum(ra.ledger.values()))
    assert ra.daily.cash.sum() == pytest.approx(ra.contribution, abs=0.05)
    assert rb.counters["items_kept"] >= ra.counters["items_kept"]


def test_engine_runs_in_the_simulator_without_overflow(bundle, start):
    e = Engine(bundle, copy.deepcopy(start.market), Flags())
    r = Simulator(start.world, 11).run(e, start.first_day, 10, copy.deepcopy(start.state))
    assert r.counters["overflow_unit_days"] == 0
    assert r.counters["units_sold"] > 0 and r.counters["units_received"] > 0
