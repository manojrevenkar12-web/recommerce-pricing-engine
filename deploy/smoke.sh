#!/usr/bin/env bash
# End-to-end check of a running API: health, readiness, auth, a buy quote, sell prices, a title,
# experiment assignment and metrics.
# Usage: EANS='["...","..."]' deploy/smoke.sh [base-url]   (keys as configured in RPE_API_KEYS)
set -euo pipefail
BASE=${1:-http://localhost:8000}
APP=${APP_KEY:-app-key}
PRICING=${PRICING_KEY:-pricing-key}
EANS=${EANS:-'["9780000000002"]'}
FIRST=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])[0])" "$EANS")

for _ in $(seq 1 90); do curl -fsS "$BASE/ready" >/dev/null 2>&1 && break; sleep 2; done
curl -fsS "$BASE/health" | grep -q ok
curl -fsS "$BASE/ready"; echo

code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/buy/quote" -H 'content-type: application/json' -d "{\"items\": $EANS}")
test "$code" = "401" || { echo "expected 401 without a key, got $code"; exit 1; }
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE/v1/sell/prices" -H "X-API-Key: $APP" -H 'content-type: application/json' \
  -d "{\"units\": [{\"unit_id\": 1, \"ean\": \"$FIRST\", \"grade\": \"good\"}]}")
test "$code" = "403" || { echo "expected 403 for the app key on sell prices, got $code"; exit 1; }

curl -fsS -X POST "$BASE/v1/buy/quote" -H "X-API-Key: $APP" -H 'content-type: application/json' -d "{\"items\": $EANS}" > quote.json
python3 - <<'PY'
import json
q = json.load(open("quote.json"))
assert q["items"] and all("decision" in i and "offer_eur" in i for i in q["items"]), q
assert abs(q["basket_total_eur"] - sum(i["offer_eur"] for i in q["items"])) < 0.011, q
known = [i for i in q["items"] if i["decision"] != "unknown_title"]
assert known, "no known title in the quote"
print("quote:", [(i["ean"], i["decision"], i["offer_eur"]) for i in q["items"]])
PY
curl -fsS -X POST "$BASE/v1/sell/prices" -H "X-API-Key: $PRICING" -H 'content-type: application/json' \
  -d "{\"units\": [{\"unit_id\": 1, \"ean\": \"$FIRST\", \"grade\": \"good\", \"days_listed\": 30, \"current_price\": 4.5}]}" > prices.json
python3 -c "import json; u=json.load(open('prices.json'))['units'][0]; assert u['allowed_range_eur'][0]-0.011 <= u['price_eur'] <= u['allowed_range_eur'][1]+0.011, u; print('price:', u['price_eur'], u['allowed_range_eur'])"
curl -fsS "$BASE/v1/titles/$FIRST" -H "X-API-Key: $PRICING" | grep -q market_price_eur
curl -fsS -X POST "$BASE/v1/experiments/buy-engine-2026-01/assign" -H "X-API-Key: $APP" -H 'content-type: application/json' -d '{"unit_ids": [1, 2, 3]}' | grep -q arms
curl -fsS "$BASE/metrics" | grep -q rpe_offers_total
echo "smoke test passed"
