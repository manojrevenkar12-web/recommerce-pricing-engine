"""Steering report and the release gate logic."""

from __future__ import annotations

import pandas as pd
import pytest

from rpe.config import load
from rpe.evaluation.gate import decide


@pytest.mark.pipeline
def test_steering_report(built, tmp_path):
    from rpe.steering.report import build

    r = build(out=tmp_path)
    assert (tmp_path / "steering.html").read_text().startswith("<!doctype html>")
    assert "## Alerts" in (tmp_path / "steering.md").read_text()
    assert r["kpis"]["revenue_eur"]["last_4_weeks"] > 0
    assert len(r["category_prices"]) == 10


def test_price_war_alert_fires():
    from rpe.steering.report import alerts

    k = pd.DataFrame(
        {
            "kept": [100] * 8,
            "offers": [200] * 8,
            "fill": [0.5] * 8,
            "gross_margin": [1000] * 8,
            "units_sold": [100] * 8,
            "stock": [1000] * 8,
        }
    )
    prices = pd.DataFrame({"category": ["fiction", "dvd"], "change_pct": [-18.0, 1.0], "titles": [500, 300]})
    a = alerts(k, prices, {"aged_180d_share": 0.1}, load("steering"))
    assert [x["what"] for x in a] == ["Price war watch: fiction"]


def _evaluation(value_pct=20.0, lo=5.0, violations=0):
    run = {"policy": "engine", "overflow_unit_days": 0, "items_bought_while_full": 0}
    sc = {
        "engine_overflow_unit_days": 10,
        "baseline_overflow_unit_days": 20,
        "engine_items_bought_while_full": 0,
        "engine_guardrail_violations": 0,
        "uplift_pct": 3.0,
    }
    return {
        "contracts": {"bad_rows_caught": 1.0, "good_rows_quarantined": 0},
        "backtest": {
            "uplift": {"value": {"pct": value_pct, "pct_lo": lo}, "pending_returns": {"pct": 2.0}},
            "runs": [run],
            "guardrails": {"prices_checked": 10, "below_min_price": violations},
            "scenarios": {"dev": [sc], "heldout": [sc]},
            "long_run": {"second_year_cash_uplift_pct": 30.0},
        },
        "models": {
            "market_price": {
                "feed_titles": {"engine": {"mean_abs_log_error": 0.07}, "rule_last_price": {"mean_abs_log_error": 0.12}},
                "cold_start_titles": {"engine": {"mean_abs_log_error": 0.3}, "rule_share_of_list_price": {"mean_abs_log_error": 0.6}},
            },
            "demand_calibration": {"overall_ratio": 1.02},
            "sellability_90d": {"expected_calibration_error": 0.03, "auc": 0.74, "auc_sales_rank_rule": 0.6},
            "price_response": {"beta_mean_abs_rel_error": 0.1},
        },
        "experiments": {
            "sell_side_price_test": {"a_a": {"false_positive_rate_by_title": 0.05}, "coverage": 1.0},
            "peeking_null": {"always_valid_reject": 0.02},
            "buy_side_offer_test": {"intervals_overlap": True},
        },
    }


def test_gate_passes_and_blocks():
    ok, lines = decide(_evaluation())
    assert ok, [x for x in lines if x["passed"] is False]
    ok, lines = decide(_evaluation(lo=-1.0))
    assert not ok and any(x["check"] == "engine_value_uplift_pct_lo" and not x["passed"] for x in lines)
    ok, _ = decide(_evaluation(violations=3))
    assert not ok
