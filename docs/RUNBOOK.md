# Runbook

## Deploy

```bash
docker build -t rpe .
docker run -p 8000:8000 -v $PWD/models:/models:ro -v $PWD/data:/data:ro \
  -e RPE_API_KEYS='{"<app key>": "app", "<pricing key>": "pricing"}' rpe
deploy/smoke.sh http://localhost:8000
```

The image serves the models in `/models` and the warehouse in `/data`. Releases are gated: deploy
models only from a pipeline run whose `rpe gate` passed (CI uploads them as the `serving` artifact).

## Watch

| metric | normal | if not |
|---|---|---|
| `rpe_offers_total{decision="buy"}` share | 30-60% of quoted items | falling sharply: check `rpe_warehouse_fill` and the shelf price; check the competitor feed arrived (market prices from the attribute model only) |
| `rpe_offers_total{decision="warehouse_full"}` | 0 | the emergency stop is on: warehouse above 98.5%; capacity planning, liquidation backlog |
| `rpe_shelf_price_eur_per_day` | 0 to 0.01 | above 0.02 for a week: buying only fast sellers; confirm with Logistics whether capacity is expected back |
| `rpe_request_seconds` p95 | < 50 ms | model files on slow storage; too many items per request (limit 200) |
| `rpe_model_age_days` | < 8 | retrain weekly: `rpe ingest && rpe warehouse && rpe train`, then `rpe eval && rpe gate` |
| 4xx share | low | key rotation; app sending malformed barcodes |

## When something looks wrong

- **A category's offers or prices jump.** Look at the steering report's market watch (`rpe steering`)
  for a competitor price move; the market price follows the feed within about a week (7-day window
  when prices move more than 25%).
- **Too much liquidation.** It is capped at 3% of stock per week. Check the shelf price and whether
  the demand model was retrained on a short or unusual history.
- **Model or data problem.** Roll back to the previous `models/` (the model card names its data
  fingerprint) and rerun `rpe gate` on it.
- **Bad source data.** Contract violations land in `data/clean/_quarantine.parquet` with a reason;
  the owner of each source is in its contract.

## Change a business setting

Edit `config/pricing.yaml` (positioning, guardrails, capacity target, long-term weight) or
`config/costs.yaml` (unit economics, owned by Finance). The API refuses to start on an invalid
pricing config; CI runs the full evaluation and gate on every change.
