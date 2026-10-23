"""`rpe gate`: reads reports/evaluation.json and decides with config/release.yaml."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpe.config import REPORTS, load


def measures(r: dict[str, Any]) -> dict[str, float | None]:
    bt, m, ex = r["backtest"], r["models"], r["experiments"]
    up = bt["uplift"]
    mp = m["market_price"]
    held = bt["scenarios"]["heldout"]
    scen = bt["scenarios"]["dev"] + held
    return {
        "contracts_bad_rows_caught": r["contracts"]["bad_rows_caught"],
        "contracts_good_rows_quarantined": r["contracts"]["good_rows_quarantined"],
        "engine_value_uplift_pct": up["value"]["pct"],
        "engine_value_uplift_pct_lo": up["value"]["pct_lo"],
        "seller_returns_vs_baseline": round(1 + up["pending_returns"]["pct"] / 100, 4),
        "overflow_unit_days": max(x["overflow_unit_days"] for x in bt["runs"] if x["policy"] == "engine"),
        "guardrail_violations": sum(v for k, v in bt["guardrails"].items() if k != "prices_checked"),
        "market_error_feed_vs_rule": round(
            mp["feed_titles"]["engine"]["mean_abs_log_error"] / mp["feed_titles"]["rule_last_price"]["mean_abs_log_error"], 4
        ),
        "market_error_cold_vs_rule": round(
            mp["cold_start_titles"]["engine"]["mean_abs_log_error"]
            / mp["cold_start_titles"]["rule_share_of_list_price"]["mean_abs_log_error"],
            4,
        ),
        "demand_calibration_ratio": m["demand_calibration"]["overall_ratio"],
        "sellability_ece": m["sellability_90d"]["expected_calibration_error"],
        "sellability_auc_gain_vs_rank": round(m["sellability_90d"]["auc"] - m["sellability_90d"]["auc_sales_rank_rule"], 4),
        "aa_false_positive_rate": ex["sell_side_price_test"]["a_a"]["false_positive_rate_by_title"],
        "sequential_false_positive_rate": ex["peeking_null"]["always_valid_reject"],
        "sell_test_ci_coverage": ex["sell_side_price_test"]["coverage"],
        "scenario_overflow_unit_days": max((x["engine_overflow_unit_days"] for x in scen), default=0),
        "scenario_guardrail_violations": sum(x["engine_guardrail_violations"] for x in scen),
        # tracked
        "heldout_scenarios_engine_ahead": round(sum(x["uplift_pct"] > 0 for x in held) / len(held), 3) if held else None,
        "surrogate_intervals_overlap": float(ex["buy_side_offer_test"]["intervals_overlap"]),
        "long_run_cash_uplift_pct": bt["long_run"].get("second_year_cash_uplift_pct"),
        "price_response_mean_abs_rel_error": m["price_response"]["beta_mean_abs_rel_error"],
    }


def decide(r: dict[str, Any], cfg: dict[str, Any] | None = None) -> tuple[bool, list[dict[str, Any]]]:
    cfg = cfg or load("release")
    ms = measures(r)
    lines = []
    ok = True
    for name, rule in cfg["blocking"].items():
        v = ms.get(name)
        passed = v is not None and ("min" not in rule or v >= rule["min"]) and ("max" not in rule or v <= rule["max"])
        ok &= passed
        bound = " and ".join(f"{k} {rule[k]}" for k in ("min", "max") if k in rule)
        lines.append({"check": name, "value": v, "bound": bound, "passed": passed, "blocking": True, "why": rule.get("why", "")})
    for name in cfg.get("tracked", []):
        lines.append({"check": name, "value": ms.get(name), "bound": "", "passed": None, "blocking": False, "why": "tracked"})
    return ok, lines


def gate(path: Path = REPORTS / "evaluation.json") -> tuple[bool, list[dict[str, Any]]]:
    r = json.loads(path.read_text())
    ok, lines = decide(r)
    r["gate"] = {"passed": ok, "checks": lines}
    path.write_text(json.dumps(r, indent=2, default=float))
    return ok, lines


def format_lines(lines: list[dict[str, Any]]) -> list[str]:
    out = []
    for x in lines:
        mark = "PASS" if x["passed"] else ("FAIL" if x["passed"] is False else "info")
        out.append(f"{mark:4}  {x['check']:36} {x['value']!s:>12}  {x['bound']}")
    return out
