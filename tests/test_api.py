"""The HTTP API (needs the generated data)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.pipeline
KEYS = {"app-key": "app", "pricing-key": "pricing"}


@pytest.fixture(scope="module")
def client(built, monkeypatch_module):
    monkeypatch_module.setenv("RPE_API_KEYS", json.dumps(KEYS))
    from rpe.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def eans(built):
    from rpe.warehouse import query

    return query("select ean from dim_title order by idx limit 40").ean.tolist()


def test_health_ready_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready").json()
    assert r["status"] == "ready" and r["titles"] == 12000
    assert "rpe_requests_total" in client.get("/metrics").text


def test_auth(client, eans):
    assert client.post("/v1/buy/quote", json={"items": eans[:2]}).status_code == 401
    assert client.post("/v1/buy/quote", json={"items": eans[:2]}, headers={"X-API-Key": "nope"}).status_code == 401
    r = client.post("/v1/sell/prices", json={"units": [{"unit_id": 1, "ean": eans[0], "grade": "good"}]}, headers={"X-API-Key": "app-key"})
    assert r.status_code == 403


def test_quote(client, eans):
    r = client.post("/v1/buy/quote", json={"items": [*eans[:25], "9780000000002"]}, headers={"X-API-Key": "app-key"})
    assert r.status_code == 200
    j = r.json()
    assert len(j["items"]) == 26
    assert j["items"][-1]["decision"] == "unknown_title"
    assert j["basket_total_eur"] == pytest.approx(sum(i["offer_eur"] for i in j["items"]), abs=0.01)
    assert j["meets_min_basket"] == (j["basket_total_eur"] >= j["min_basket_eur"])
    for it in j["items"][:-1]:
        assert (it["offer_eur"] > 0) == (it["decision"] == "buy")
        assert it["message"]


def test_quote_validation(client):
    assert client.post("/v1/buy/quote", json={"items": []}, headers={"X-API-Key": "app-key"}).status_code == 422


def test_prices_and_title(client, eans):
    body = {
        "units": [
            {"unit_id": 1, "ean": eans[3], "grade": "good", "days_listed": 30, "current_price": 6.0},
            {"unit_id": 2, "ean": "1234", "grade": "like_new"},
        ]
    }
    j = client.post("/v1/sell/prices", json=body, headers={"X-API-Key": "pricing-key"}).json()
    u = j["units"][0]
    assert u["allowed_range_eur"][0] - 0.01 <= u["price_eur"] <= u["allowed_range_eur"][1] + 0.01
    assert j["units"][1]["error"] == "unknown_title"
    t = client.get(f"/v1/titles/{eans[3]}", headers={"X-API-Key": "pricing-key"}).json()
    assert t["market_price_p10_p90_eur"][0] <= t["market_price_eur"] <= t["market_price_p10_p90_eur"][1]
    assert client.get("/v1/titles/0000000000000", headers={"X-API-Key": "pricing-key"}).status_code == 404


def test_experiment_assignment_is_stable(client):
    a = client.post(
        "/v1/experiments/sell-engine-2026-01/assign", json={"unit_ids": list(range(200))}, headers={"X-API-Key": "app-key"}
    ).json()
    b = client.post(
        "/v1/experiments/sell-engine-2026-01/assign", json={"unit_ids": list(range(200))}, headers={"X-API-Key": "pricing-key"}
    ).json()
    assert a["arms"] == b["arms"] and 60 < a["arms"].count("treatment") < 140
    assert client.post("/v1/experiments/nope/assign", json={"unit_ids": [1]}, headers={"X-API-Key": "app-key"}).status_code == 404
