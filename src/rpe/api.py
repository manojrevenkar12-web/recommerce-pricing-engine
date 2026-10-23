"""HTTP API: buy quotes for the seller app, sell prices for the repricer, title lookups for
Pricing, experiment assignment, and health and metrics for operations.

Callers authenticate with an API key (header `X-API-Key`); keys and their roles come from the
environment variable RPE_API_KEYS as JSON, e.g. {"app-key": "app", "pricing-key": "pricing"}.
The seller app ("app") may only ask for buy quotes; Pricing ("pricing") may call everything.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from rpe import metrics as M
from rpe.config import REPORTS, load
from rpe.experiments.assign import arm

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rpe.service import PricingService

    if "service" not in STATE:
        STATE["service"] = PricingService.from_warehouse()
    svc = STATE["service"]
    M.MODEL_AGE.set(svc.info()["model_age_days"])
    yield


app = FastAPI(title="recommerce-pricing-engine", version="0.1.0", lifespan=lifespan)


def _keys() -> dict[str, str]:
    raw = os.environ.get("RPE_API_KEYS", "")
    return json.loads(raw) if raw else {}


def role(x_api_key: str | None = Header(default=None)) -> str:
    keys = _keys()
    if not keys:
        raise HTTPException(503, "no API keys configured (RPE_API_KEYS)")
    if x_api_key is None or x_api_key not in keys:
        raise HTTPException(401, "missing or unknown API key")
    return keys[x_api_key]


def need(*roles: str):
    def check(r: str = Depends(role)) -> str:
        if r not in roles:
            raise HTTPException(403, f"role '{r}' may not call this")
        return r

    return check


@app.middleware("http")
async def observe(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    name = getattr(route, "path", request.url.path)
    M.REQUESTS.labels(name, str(response.status_code)).inc()
    M.LATENCY.labels(name).observe(time.perf_counter() - t)
    return response


class QuoteRequest(BaseModel):
    items: list[str] = Field(..., min_length=1, max_length=200, description="EAN-13 / ISBN-13 barcodes scanned by the seller")
    returning_seller: bool = False


class UnitIn(BaseModel):
    unit_id: int
    ean: str
    grade: Literal["like_new", "very_good", "good", "acceptable"]
    days_listed: int = Field(0, ge=0, le=2000)
    current_price: float | None = Field(None, gt=0, le=2000)


class PriceRequest(BaseModel):
    units: list[UnitIn] = Field(..., min_length=1, max_length=5000)


class AssignRequest(BaseModel):
    unit_ids: list[int] = Field(..., min_length=1, max_length=100_000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    svc = STATE.get("service")
    if svc is None:
        raise HTTPException(503, "models not loaded")
    return {"status": "ready", **svc.info()}


@app.get("/metrics")
def prometheus() -> Response:
    return Response(generate_latest(M.REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/buy/quote")
def quote(req: QuoteRequest, _: str = Depends(need("app", "pricing"))) -> dict[str, Any]:
    svc = STATE["service"]
    out = svc.quote(req.items, req.returning_seller)
    for it in out["items"]:
        M.OFFERS.labels(it["decision"]).inc()
        M.OFFER_EUR.inc(it["offer_eur"])
    M.SHELF_PRICE.set(out["shelf_price_eur_per_day"])
    M.FILL.set(out["warehouse_fill"])
    return out


@app.post("/v1/sell/prices")
def prices(req: PriceRequest, _: str = Depends(need("pricing"))) -> dict[str, Any]:
    svc = STATE["service"]
    rows = svc.prices([u.model_dump() for u in req.units])
    for r in rows:
        M.PRICES.labels("error" if "error" in r else ("liquidate" if r["liquidate"] else "priced")).inc()
    return {"as_of": svc.info()["as_of"], "units": rows}


@app.get("/v1/titles/{ean}")
def title(ean: str, _: str = Depends(need("pricing"))) -> dict[str, Any]:
    t = STATE["service"].title(ean)
    if t is None:
        raise HTTPException(404, "unknown title")
    return t


@app.get("/v1/experiments")
def experiments(_: str = Depends(need("pricing"))) -> list[dict[str, Any]]:
    return load("experiments")


@app.post("/v1/experiments/{exp_id}/assign")
def assign(exp_id: str, req: AssignRequest, _: str = Depends(need("app", "pricing"))) -> dict[str, Any]:
    exp = next((e for e in load("experiments") if e["id"] == exp_id), None)
    if exp is None:
        raise HTTPException(404, "unknown experiment")
    a = arm(np.array(req.unit_ids), exp["salt"], exp["treatment_share"])
    return {"experiment": exp_id, "unit": exp["unit"], "arms": ["treatment" if x else "control" for x in a]}


@app.get("/v1/steering")
def steering(_: str = Depends(need("pricing"))) -> dict[str, Any]:
    p = REPORTS / "steering.json"
    if not p.exists():
        raise HTTPException(404, "no steering report yet (rpe steering)")
    return json.loads(p.read_text())
