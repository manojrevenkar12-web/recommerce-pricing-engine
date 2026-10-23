"""`rpe eval`: every measurement behind the release gate, written to reports/evaluation.json."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rpe.config import CLEAN, REPORTS, ROOT, TRUTH, load
from rpe.evaluation.backtest import load_start, paired_uplift, run_many, save_json
from rpe.evaluation.experiments import evaluate_experiments
from rpe.evaluation.models import evaluate_models

ABLATIONS = ["engine_sell_only", "engine_buy_only", "engine_no_market_model", "engine_no_capacity", "engine_no_long_term"]


def scenarios(kind: str) -> list[dict[str, Any]]:
    return [yaml.safe_load(p.read_text()) for p in sorted((ROOT / "scenarios" / kind).glob("*.yaml"))]


def contracts_check(clean: Path = CLEAN, truth: Path = TRUTH) -> dict[str, Any]:
    injected = json.loads((truth / "history.json").read_text())["injected_bad_rows"]
    rep = json.loads((clean / "_ingest_report.json").read_text())
    q = {s["source"]: s["quarantined"] for s in rep["sources"]}
    caught = {k: min(1.0, q.get(k, 0) / v) for k, v in injected.items() if v}
    lost = sum(max(0, q.get(k, 0) - injected.get(k, 0)) for k in q)
    return {"injected": injected, "quarantined": q, "bad_rows_caught": round(min(caught.values()), 4), "good_rows_quarantined": int(lost)}


def evaluate(workers: int | None = None, out: Path = REPORTS) -> dict[str, Any]:
    t0 = time.time()
    from rpe.train import Bundle

    bundle = Bundle.load()
    s = load_start(bundle)
    cfg = load("release")
    bt = cfg["backtest"]
    dev, held = scenarios("dev"), scenarios("heldout")

    specs: list[dict[str, Any]] = []
    for sd in bt["seeds"]:
        specs += [{"name": "baseline", "seed": sd}, {"name": "engine", "seed": sd, "audit": True}]
    for sd in bt["ablation_seeds"]:
        specs += [{"name": n, "seed": sd} for n in ABLATIONS]
    specs += [{"name": n, "seed": bt["long_run_seed"], "days": bt["long_run_days"], "tag": "long"} for n in ("baseline", "engine")]
    for kind, scs in (("dev", dev), ("heldout", held)):
        for sc in scs:
            specs += [
                {"name": "baseline", "seed": 501, "scenario": sc, "tag": kind},
                {"name": "engine", "seed": 501, "scenario": sc, "audit": True, "tag": kind},
            ]
    tags = [sp.pop("tag", "main") for sp in specs]
    res = run_many(specs, workers, s)
    t_bt = time.time() - t0

    rows = pd.DataFrame([r["row"] for r in res])
    rows["tag"] = tags
    main = rows[(rows.tag == "main") & rows.policy.isin(["baseline", "engine"])]
    abl = rows[rows.policy.isin(ABLATIONS) & (rows.tag == "main")]
    violations = [r.get("violations", {}) for r, t in zip(res, tags, strict=True) if t == "main" and r["row"]["policy"] == "engine"]

    up = {
        m: paired_uplift(main, m, "engine")
        for m in (
            "value",
            "cash",
            "value_incl_sellers",
            "revenue",
            "revenue_minus_purchases",
            "pending_returns",
            "units_sold",
            "units_bought",
        )
    }
    headline = {
        "seeds": len(bt["seeds"]),
        "value_pct": up["value"]["pct"],
        "value_pct_lo": up["value"]["pct_lo"],
        "value_pct_hi": up["value"]["pct_hi"],
        "value_eur": up["value"]["diff"],
        "cash_pct": up["cash"]["pct"],
        "returns_pct": up["pending_returns"]["pct"],
        "value_incl_sellers_pct": up["value_incl_sellers"]["pct"],
    }
    full = main[main.policy == "engine"].set_index("seed")
    ablations = {}
    for n in ABLATIONS:
        a = abl[abl.policy == n].set_index("seed")
        seeds = a.index.intersection(full.index)
        d = (full.loc[seeds, "value"] - a.loc[seeds, "value"]).to_numpy()
        d2 = (full.loc[seeds, "value_incl_sellers"] - a.loc[seeds, "value_incl_sellers"]).to_numpy()
        half = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        ablations[n] = {
            "value_of_part": round(float(d.mean()), 2),
            "lo": round(float(d.mean() - half), 2),
            "hi": round(float(d.mean() + half), 2),
            "value_of_part_incl_sellers": round(float(d2.mean()), 2),
            "seeds": len(d),
        }

    long = [r for r, t in zip(res, tags, strict=True) if t == "long"]
    long_run: dict[str, Any] = {}
    for r in long:
        d = r["daily"]
        half = (d.day - d.day.min()) // 182
        long_run[r["row"]["policy"]] = {
            "cash_by_half_year": d.groupby(half).cash.sum().round(0).tolist(),
            "value": r["row"]["value"],
            "cash": r["row"]["cash"],
        }
    if {"baseline", "engine"} <= set(long_run):
        e, b = long_run["engine"]["cash_by_half_year"], long_run["baseline"]["cash_by_half_year"]
        long_run["second_year_cash_uplift_pct"] = round(100 * (sum(e[2:]) / sum(b[2:]) - 1), 2) if len(e) > 2 else None

    def scen(kind: str) -> list[dict[str, Any]]:
        out = []
        for r, t in zip(res, tags, strict=True):
            if t != kind:
                continue
            row = r["row"]
            out.append(
                {
                    **{
                        k: row[k]
                        for k in (
                            "scenario",
                            "policy",
                            "value",
                            "cash",
                            "pending_returns",
                            "max_fill",
                            "overflow_unit_days",
                            "items_bought_while_full",
                            "units_bought",
                            "units_sold",
                            "units_liquidated",
                        )
                    },
                    "violations": r.get("violations", {}),
                }
            )
        df = pd.DataFrame(out)
        summary = []
        for name, g in df.groupby("scenario", sort=False):
            b = g[g.policy == "baseline"].iloc[0]
            e = g[g.policy == "engine"].iloc[0]
            summary.append(
                {
                    "scenario": name,
                    "baseline_value": b.value,
                    "engine_value": e.value,
                    "uplift_pct": round(100 * (e.value / b.value - 1), 2),
                    "engine_cash": e.cash,
                    "baseline_cash": b.cash,
                    "engine_returns_vs_baseline": round(e.pending_returns / max(b.pending_returns, 1), 3),
                    "engine_max_fill": e.max_fill,
                    "engine_overflow_unit_days": int(e.overflow_unit_days),
                    "baseline_overflow_unit_days": int(b.overflow_unit_days),
                    "engine_items_bought_while_full": int(e.items_bought_while_full),
                    "baseline_items_bought_while_full": int(b.items_bought_while_full),
                    "engine_guardrail_violations": int(sum(v for k, v in e.violations.items() if k != "prices_checked")),
                }
            )
        return summary

    scen_dev, scen_held = scen("dev"), scen("heldout")
    first_run = record_first_run(scen_held)

    t1 = time.time()
    models = evaluate_models(s)
    t_models = time.time() - t1
    t1 = time.time()
    exps = evaluate_experiments(s, cfg["experiments"]["sell_test_seeds"])
    t_exp = time.time() - t1

    api_path = out / "api_latency.json"
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_card": {
            k: bundle.card[k]
            for k in (
                "trained_at",
                "history",
                "data_fingerprint",
                "rows",
                "price_response",
                "condition_factors",
                "acceptance",
                "seller_value",
                "items_per_parcel",
            )
        },
        "contracts": contracts_check(),
        "models": models,
        "backtest": {
            "horizon_days": int(s.world.cfg["calendar"]["forward_days"]),
            "headline": headline,
            "uplift": up,
            "runs": main.drop(columns=["ledger"]).to_dict("records"),
            "ledgers": {f"{r.policy}:{r.seed}": r.ledger for r in main.itertuples()},
            "ablations": ablations,
            "ablation_seeds": bt["ablation_seeds"],
            "long_run": long_run,
            "guardrails": _sum_violations(violations),
            "scenarios": {"dev": scen_dev, "heldout": scen_held, "heldout_first_run": first_run},
        },
        "experiments": exps,
        "api": json.loads(api_path.read_text()) if api_path.exists() else None,
        "seconds": {
            "backtests": round(t_bt, 1),
            "models": round(t_models, 1),
            "experiments": round(t_exp, 1),
            "total": round(time.time() - t0, 1),
        },
    }
    save_json(report, out / "evaluation.json")
    return report


def _sum_violations(vs: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in vs:
        for k, n in v.items():
            out[k] = out.get(k, 0) + int(n)
    return out


def record_first_run(summary: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first time the held-out scenarios run, their results are written down for good: later
    changes may improve on them, but the first-run numbers are what the engine did on unseen ground."""
    path = ROOT / "scenarios" / "heldout" / "FIRST_RUN.md"
    data = ROOT / "scenarios" / "heldout" / "first_run.json"
    if not summary:
        return None
    if data.exists():
        return json.loads(data.read_text())
    rec = {"recorded_at": datetime.now(UTC).isoformat(timespec="seconds"), "results": summary}
    data.write_text(json.dumps(rec, indent=2, default=float))
    L = [
        "# Held-out scenarios: first run",
        "",
        f"Recorded automatically by `rpe eval` on {rec['recorded_at']}, the first time these scenarios were run. "
        "They were written by someone who had not seen the engine, its code or its results. This file is not updated afterwards.",
        "",
        "| scenario | rules: value | engine: value | engine vs rules | engine max fill | overflow (engine / rules) | guardrail violations |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        L.append(
            f"| {r['scenario']} | {r['baseline_value']:,.0f} | {r['engine_value']:,.0f} | {r['uplift_pct']:+.1f}% | {r['engine_max_fill']:.1%} | "
            f"{r['engine_overflow_unit_days']:,} / {r['baseline_overflow_unit_days']:,} | {r['engine_guardrail_violations']} |"
        )
    path.write_text("\n".join(L) + "\n")
    return rec
