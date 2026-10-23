"""Command line: `rpe run` rebuilds everything and ends with the release gate."""

from __future__ import annotations

import argparse
import json
import sys
import time


def _step(name: str):
    print(f"\n== {name}", flush=True)
    return time.time()


def cmd_history(_: argparse.Namespace) -> int:
    from rpe.sim.history import generate

    t = _step("simulate 2025 under the rule-based pricing and export the raw data")
    print(json.dumps(generate()), f"({time.time() - t:.0f}s)")
    return 0


def cmd_ingest(_: argparse.Namespace) -> int:
    from rpe.contracts import ingest

    t = _step("ingest raw exports through the data contracts")
    r = ingest()
    for s in r["sources"]:
        print(
            f"  {s['source']:20} read {s['rows_read']:>9,}  loaded {s['loaded']:>9,}  quarantined {s['quarantined']:>6,}  {s['reasons'] or ''}"
        )
    print(f"({time.time() - t:.0f}s)")
    return 0


def cmd_warehouse(_: argparse.Namespace) -> int:
    from rpe.warehouse import build

    t = _step("build the warehouse")
    print(json.dumps(build()), f"({time.time() - t:.0f}s)")
    return 0


def cmd_train(_: argparse.Namespace) -> int:
    from rpe.train import train

    t = _step("train market price, demand, acceptance, grading and seller-value models")
    b = train()
    c = b.card
    print(
        json.dumps({k: c[k] for k in ("history", "rows", "price_response", "condition_factors", "items_per_parcel", "timings")}, indent=1),
        f"({time.time() - t:.0f}s)",
    )
    return 0


def cmd_eval(a: argparse.Namespace) -> int:
    from rpe.evaluation.run import evaluate

    t = _step("evaluate: backtests, ablations, scenarios, models, experiments")
    r = evaluate(a.workers)
    print(json.dumps(r["backtest"]["headline"]), f"({time.time() - t:.0f}s)")
    return 0


def cmd_bench_api(a: argparse.Namespace) -> int:
    import os

    import numpy as np
    from fastapi.testclient import TestClient

    from rpe.config import REPORTS
    from rpe.warehouse import query

    os.environ.setdefault("RPE_API_KEYS", json.dumps({"bench": "app"}))
    key = next(iter(json.loads(os.environ["RPE_API_KEYS"])))
    from rpe.api import app

    eans = query("select ean from dim_title using sample 400 rows (reservoir, 5)").ean.tolist()
    ts = []
    with TestClient(app) as c:
        for k in range(a.requests):
            body = {"items": eans[(k * a.items) % 375 : (k * a.items) % 375 + a.items]}
            t = time.perf_counter()
            r = c.post("/v1/buy/quote", json=body, headers={"X-API-Key": key})
            ts.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200, r.text
    out = {
        "requests": a.requests,
        "items": a.items,
        "p50_ms": round(float(np.percentile(ts, 50)), 2),
        "p95_ms": round(float(np.percentile(ts, 95)), 2),
        "max_ms": round(max(ts), 2),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "api_latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    return 0


def cmd_steering(_: argparse.Namespace) -> int:
    from rpe.steering.report import build

    t = _step("weekly steering report")
    r = build()
    print(f"  {len(r['alerts'])} alerts; reports/steering.html, steering.md ({time.time() - t:.0f}s)")
    return 0


def cmd_report(_: argparse.Namespace) -> int:
    from rpe.evaluation.report import write

    write()
    print("  reports/EVALUATION.md")
    return 0


def cmd_gate(_: argparse.Namespace) -> int:
    from rpe.evaluation.gate import format_lines, gate

    _step("release gate")
    ok, lines = gate()
    print("\n".join(format_lines(lines)))
    print("\nRELEASE GATE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_run(a: argparse.Namespace) -> int:
    for f in (cmd_history, cmd_ingest, cmd_warehouse, cmd_train, cmd_bench_api, cmd_eval, cmd_steering, cmd_gate, cmd_report):
        rc = f(a)
        if rc and f is not cmd_gate:
            return rc
        if f is cmd_gate:
            gate_rc = rc
    return gate_rc


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("rpe.api:app", host=a.host, port=a.port, workers=1)
    return 0


def cmd_quote(a: argparse.Namespace) -> int:
    from rpe.service import PricingService

    svc = PricingService.from_warehouse()
    print(json.dumps(svc.quote(a.eans, a.returning), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rpe", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, f, h in [
        ("history", cmd_history, "simulate and export the 2025 history"),
        ("ingest", cmd_ingest, "apply the data contracts"),
        ("warehouse", cmd_warehouse, "build the DuckDB warehouse"),
        ("train", cmd_train, "train all models"),
        ("eval", cmd_eval, "run every evaluation"),
        ("bench-api", cmd_bench_api, "measure API latency"),
        ("steering", cmd_steering, "write the weekly steering report"),
        ("report", cmd_report, "write reports/EVALUATION.md"),
        ("gate", cmd_gate, "apply the release gate"),
        ("run", cmd_run, "all of the above, ending with the gate"),
        ("serve", cmd_serve, "start the API"),
        ("quote", cmd_quote, "buy quote for barcodes"),
    ]:
        sp = sub.add_parser(name, help=h)
        sp.set_defaults(func=f)
        if name in ("eval", "run"):
            sp.add_argument("--workers", type=int, default=None, help="parallel backtest processes (default RPE_WORKERS or 1)")
        if name in ("bench-api", "run"):
            sp.add_argument("--requests", type=int, default=200)
            sp.add_argument("--items", type=int, default=25)
        if name == "serve":
            sp.add_argument("--host", default="0.0.0.0")
            sp.add_argument("--port", type=int, default=8000)
        if name == "quote":
            sp.add_argument("eans", nargs="+")
            sp.add_argument("--returning", action="store_true")
    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    sys.exit(main())
