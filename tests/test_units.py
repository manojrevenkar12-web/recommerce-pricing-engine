"""Unit tests: no generated data needed."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from rpe import ean
from rpe.config import costs, load
from rpe.engine.core import CapacityController, hold_value, net_sale, queue_value
from rpe.engine.rules import SellBounds, _intersect, price_interval, sell_bounds, sell_violations, validate
from rpe.experiments.analysis import cuped, delta_ratio, sample_size, srm, welch
from rpe.experiments.assign import arm
from rpe.experiments.sequential import always_valid_p, peeking_study
from rpe.hashing import hash_uniform
from rpe.market import MarketView
from rpe.models.market_price import snapshot
from rpe.policy import MarketUpdate, Warehouse


def test_ean_check_digits():
    assert ean.valid("9783161484100")
    assert not ean.valid("9783161484101")
    assert not ean.valid("978316148410")
    code = ean.make("978", 12345)
    assert ean.valid(code) and code.startswith("978")


def test_hash_is_deterministic_and_uniform():
    a = hash_uniform(np.arange(100_000), 7)
    assert np.array_equal(a, hash_uniform(np.arange(100_000), 7))
    assert abs(a.mean() - 0.5) < 0.01
    assert not np.array_equal(a, hash_uniform(np.arange(100_000), 8))


def test_assignment_share_and_stability():
    a = arm(np.arange(50_000), 42, 0.3)
    assert abs(a.mean() - 0.3) < 0.01
    assert np.array_equal(a[:10], arm(np.arange(10), 42, 0.3))


def test_welch_and_cuped():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4000)
    y = x + rng.normal(0, 0.5, 4000)
    t = rng.random(4000) < 0.5
    y = y + 0.2 * t
    e = welch(y, t)
    assert e.lo < 0.2 < e.hi
    c, removed = cuped(y, x, t)
    assert c.lo < 0.2 < c.hi
    assert removed > 0.7 and (c.hi - c.lo) < (e.hi - e.lo) / 2


def test_delta_ratio_matches_simulation():
    rng = np.random.default_rng(1)
    n = rng.poisson(5, 3000).astype(float) + 1
    num = n * (2 + rng.normal(0, 0.5, 3000))
    t = rng.random(3000) < 0.5
    e = delta_ratio(num, n, t)
    assert e.lo < 0 < e.hi


def test_srm_and_sample_size():
    assert srm(5000, 5000) > 0.9
    assert srm(5300, 4700) < 0.001
    assert sample_size(1.0, 0.1) == pytest.approx(3140, rel=0.01)


def test_peeking_inflates_and_msprt_controls():
    r = peeking_study(400, 20, 50, seed=3)
    assert r["naive_reject_any_look"] > 0.12
    assert r["always_valid_reject"] <= 0.05


def test_always_valid_p_is_monotone():
    p = always_valid_p(np.array([0.0, 0.1, 0.05, 0.2]), np.array([0.04, 0.02, 0.01, 0.005]), 0.01)
    assert np.all(np.diff(p) <= 1e-12)


def test_hold_value_limits():
    c = costs()
    p = np.array([5.0])
    v_fast, ps, _ = hold_value(p, np.array([10.0]), np.array([365.0]), 0.004, 0.05, c)
    assert ps[0] == pytest.approx(1.0) and v_fast[0] == pytest.approx(net_sale(p, c)[0], abs=0.01)
    v_never, ps0, t0 = hold_value(p, np.array([1e-12]), np.array([100.0]), 0.004, 0.05, c)
    assert ps0[0] < 1e-6 and t0[0] == pytest.approx(100.0) and v_never[0] == pytest.approx(0.05 - 0.4, abs=1e-3)


def test_queue_value_first_copy_equals_single_copy():
    c = costs()
    p, rate, W = np.array([6.0]), np.array([0.03]), np.array([365.0])
    a = queue_value(p, rate, np.array([1.0]), W, 0.004, 0.05, c)
    b = hold_value(p, rate, W, 0.004, 0.05, c)
    for x, y in zip(a, b, strict=True):
        assert x[0] == pytest.approx(y[0], rel=1e-6)


def test_extra_copies_are_worth_less():
    c = costs()
    vals = [
        queue_value(np.array([6.0]), np.array([0.02]), np.array([k]), np.array([365.0]), 0.004, 0.05, c)[0][0]
        for k in (1.0, 3.0, 8.0, 20.0)
    ]
    assert all(a > b for a, b in itertools.pairwise(vals))


def test_interval_priorities():
    lo, hi = _intersect(np.array([1.0, 1.0, 1.0]), np.array([5.0, 5.0, 5.0]), np.array([2.0, 6.0, -1.0]), np.array([3.0, 8.0, 0.5]))
    assert lo.tolist() == [2.0, 5.0, 1.0] and hi.tolist() == [3.0, 5.0, 1.0]


def test_price_interval_respects_min_price_and_weekly_change():
    cfg, c = load("pricing"), costs()
    b = SellBounds(np.array([0.45, 0.45]), np.array([1.8, 1.8]), [[], []])
    lo, hi = price_interval(cfg, c, np.array([10.0, 0.5]), np.array([10.0, np.nan]), np.array([np.nan, np.nan]), b)
    assert lo[0] == pytest.approx(7.5) and hi[0] == pytest.approx(12.5)
    assert lo[1] == pytest.approx(c["sell"]["min_price"])  # the market says 0.5, the floor wins


def test_violation_checker_flags_bad_prices():
    cfg, c = load("pricing"), costs()
    b = SellBounds(np.full(3, 0.45), np.full(3, 1.8), [[], [], []])
    v = sell_violations(
        cfg,
        c,
        np.array([0.99, 30.0, 10.0]),
        np.array([np.nan, np.nan, 5.0]),
        np.array([5.0, 5.0, 8.0]),
        np.array([np.nan, np.nan, np.nan]),
        b,
    )
    assert v["below_min_price"] == 1 and v["outside_steering_band"] >= 1 and v["weekly_change_too_large"] == 1


def test_steering_rules_apply_strictest_bound():
    cfg = load("pricing")
    from rpe.catalog import CATEGORIES, FORMATS

    b = sell_bounds(
        cfg,
        np.array([CATEGORIES.index("textbooks"), CATEGORIES.index("textbooks")]),
        np.array([FORMATS.index("paperback")] * 2),
        np.array([1, 3]),
    )
    assert b.max_index[0] == 1.5 and b.max_index[1] == 1.1
    assert "acceptable copies stay cheap" in b.rules[1]


def test_pricing_config_is_valid_and_bad_config_is_caught():
    assert validate(load("pricing")) == []
    bad = {**load("pricing"), "sell": {**load("pricing")["sell"], "min_index": 2.0}}
    assert validate(bad)
    bad2 = {**load("pricing"), "steering": [{"name": "x", "match": {"colour": "red"}, "sell": {"max_index": 1.2}}]}
    assert validate(bad2)


def test_capacity_price_rises_when_filling_and_falls_after():
    cfg = load("pricing")["capacity"]
    cc = CapacityController(cfg)
    depth = np.zeros(3)
    for d in range(30):
        cc.update(d, Warehouse(9500 + 20 * d, 300, 10_000, depth))
    high = cc.price
    assert high > 0
    for d in range(30, 90):
        cc.update(d, Warehouse(7000, 100, 10_000, depth))
    assert cc.price < high
    assert 0 <= cc.price <= cfg["max_price"]


def test_feed_smoothing_ignores_dumping_and_follows_moves():
    mv = MarketView(2)
    for d in range(28):
        price = np.array([5.0, 8.0 if d < 20 else 4.0])
        if d == 10:
            price[0] = 1.0  # one dumping offer
        mv.observe(MarketUpdate(d, np.array([0, 1]), price, np.array([3, 3])))
    s = snapshot(mv, 27)
    assert np.exp(s.feed[0]) == pytest.approx(5.0, rel=0.01)
    assert np.exp(s.feed[1]) == pytest.approx(4.0, rel=0.01)  # moved: recent week wins


def test_contract_check_quarantines_with_reasons():
    from rpe.contracts import Contract, check

    c = Contract(
        "t",
        "o",
        "d",
        ["id"],
        {
            "id": {"type": "int", "required": True, "min": 1},
            "ean": {"type": "string", "required": True, "rule": "ean13"},
            "p": {"type": "float", "required": True, "min_exclusive": 0},
        },
    )
    df = pd.DataFrame({"id": [1, 2, 2, 0], "ean": ["9783161484100", "9783161484100", "9783161484100", "123"], "p": [1.0, -1.0, 2.0, 1.0]})
    good, bad = check(df, c, {})
    assert good.id.tolist() == [1, 2]
    assert sorted(bad.reason.tolist()) == ["id: below 1", "p: not above 0"]
