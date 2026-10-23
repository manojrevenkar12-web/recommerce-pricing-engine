"""Prometheus metrics for the pricing API."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()
REQUESTS = Counter("rpe_requests_total", "API requests", ["endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram(
    "rpe_request_seconds", "API latency", ["endpoint"], registry=REGISTRY, buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)
)
OFFERS = Counter("rpe_offers_total", "Items quoted, by decision", ["decision"], registry=REGISTRY)
OFFER_EUR = Counter("rpe_offer_eur_total", "Sum of offers shown", registry=REGISTRY)
PRICES = Counter("rpe_prices_total", "Units priced, by outcome", ["outcome"], registry=REGISTRY)
SHELF_PRICE = Gauge("rpe_shelf_price_eur_per_day", "Current shadow price of a warehouse slot", registry=REGISTRY)
FILL = Gauge("rpe_warehouse_fill", "Stock and parcels on the way over capacity", registry=REGISTRY)
MODEL_AGE = Gauge("rpe_model_age_days", "Days since the models were trained", registry=REGISTRY)
