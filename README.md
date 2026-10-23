# recommerce-pricing-engine

**Pricing and AI for a second-hand books and media business: what to offer sellers for each
scanned item (or not to buy it), what to charge for every copy on the shelf and when to liquidate
it, with the warehouse, experiments and long-term seller behaviour in the same decisions. Built as a
production service with models, an experimentation framework, a steering report for business-unit
management and a release gate.**

The business buys used books, CDs, DVDs, Blu-rays and games from consumers through an app (the
seller scans barcodes, sees an offer per item and sends a parcel if the basket reaches 10 EUR), grades
what arrives, stores it, and sells it online. Pricing decides both sides. The market, the titles and
the sellers are simulated; the pricing, the models and the engineering are the real thing.

| Decision | Today's rules | This engine |
|---|---|---|
| Market price of a title | the last competitor price; 40% of list price if there is none | robust 28/7-day feed price that ignores dumping offers and follows real moves; a cold-start model with a P10-P90 range for titles nobody else sells |
| Buy offer, buy / no-buy | share of the competitor price by sales-rank bucket | the offer that maximises P(seller keeps it) x (value of the copy that will arrive - offer), counting grades, rejects, handling, the parcel, copies already held, the shelf price and the long-term value of a satisfied seller |
| Sell price, markdown, liquidation | one cent under the lowest competitor, -15% after 120 days, -30% after 240, liquidate at 365 | the price that maximises expected value over the copy's remaining shelf life (demand model, price response, holding cost), repriced weekly; liquidate when keeping is worth less than bulk |
| Warehouse | stop buying at 97% full | a shelf price per slot-day that rises with projected fill, felt by both sides |
| Positioning by brand, category, attribute | none | steering rules in [config/pricing.yaml](config/pricing.yaml), applied before optimisation, every decision says which bound it |

## Results

From `rpe eval` ([reports/EVALUATION.md](reports/EVALUATION.md)); CI reruns everything on each push
and `rpe gate` blocks a release when a blocking check fails. Policies are scored on a simulated
market whose truth the models never see: they learn only from the exported 2025 history.

**Engine against today's rules**, six months forward from the end of 2025, five market replays
(same sellers, same market):

| | rules | engine | difference (95% interval) |
|---|---|---|---|
| Value: cash contribution + value of stock held at the end | 144,987 EUR | 198,645 EUR | **+37.0%** (+35.8% to +38.3%) |
| Cash contribution within the six months | 60,867 EUR | 46,454 EUR | -23.7% (-27.9% to -19.5%) |
| Sellers who will come back | 1,228 | 2,154 | +75.4% |
| Revenue | 286,670 EUR | 323,340 EUR | +12.8% |
| Marketing cost per new seller who sends a parcel | 2.17 EUR | 1.54 EUR | -29% |

The engine pays sellers more for titles that sell, buys fewer duplicates and slow sellers, prices
above the competitor where demand allows and liquidates copies not worth their shelf space. That
first costs cash (better stock is bought before it is sold). **Over two years the engine earns
+46% more cash in the second year** (one replay: per half-year 60k / 77k / 53k / 65k EUR for the rules,
48k / 109k / 81k / 91k for the engine).

**On scenarios it had never seen.** Five stress scenarios were written by someone who had seen
neither the engine nor its results, and run once ([first run](scenarios/heldout/FIRST_RUN.md)):

| held-out scenario | engine vs rules (value) |
|---|---|
| a competitor closes, then returns with a games price war | +38.2% |
| a slow recession: demand -25%, prices -15%, more sellers, 15% less space | +35.0% |
| 35% of the warehouse closed during a strong spring clear-out, CDs/DVDs fading | +49.4% |
| a curriculum change halves textbook demand overnight | +32.3% |
| a TV adaptation doubles fiction demand for two months, then snaps back | +34.3% |

In the warehouse squeeze both policies start with more stock than the remaining shelves (116%):
the engine stopped buying until it was back under capacity and had a third less overflow than
the rules (32k vs 48k unit-days), but it did overflow. See [what changed](#what-changed-after-the-first-held-out-run).

**What each part is worth** (engine with vs without it, six months, two replays): this is the
AI roadmap ranked by measured value ([docs/ROADMAP.md](docs/ROADMAP.md)).

| part | value | |
|---|---|---|
| Warehouse shelf price (capacity-aware buying and selling) | +30.7k EUR | |
| Buy offers and buy/no-buy (sellability, acceptance, copies held) | +26.3k EUR | |
| Market price definition (robust feed price, cold start) | +19.5k EUR | |
| Dynamic sell prices and liquidation | +18.3k EUR | |
| Long-term seller value in offers | -5.3k EUR; +0.3k counting returning sellers at 10 EUR | a business choice, see the roadmap |

**Models against the truth** (the simulator knows it; the models never saw it):

- Market price: mean absolute log error 0.075 for titles with competitor offers (last competitor
  price: 0.121) and 0.30 for cold-start titles (40% of list price: 0.66).
- Price response, recovered from the randomised price tests: within 10% of the true elasticity on
  average (largest miss 22%, textbooks); condition price factors to within 0.01; copies sharing a
  title's buyers: power 0.92 (true 1.0).
- Sellability (P(sold within 90 days)): AUC 0.74 against 0.60 for the sales rank the rules use;
  calibration error 0.045 against 90 days of simulated sales.
- Seller acceptance: mean absolute error 0.076 against the true probability of keeping an item.

**Does the experiment framework tell the truth?**

- A/A tests analysed by title (the unit of randomisation) claim a difference 4.3% of the time at
  5%; analysed copy by copy, as if copies were independent, 54%.
- Checked daily for four weeks, a fixed-horizon test stopped at the first p < 0.05 is wrong 25.8% of
  the time; the always-valid p-value (mSPRT) 1.3%, and it still detects a real effect of 0.1 sd in
  82% of tests.
- A sell-side price test (engine vs rule prices, randomised by title, 13 weeks) contained the true
  effect in 4 of 4 replays; CUPED removed 34% of its variance. **The true effect was negative**
  (-0.4 to -0.7 EUR per title): over 13 weeks, with unsold copies counted at a conservative book
  value, the engine's patience is not rewarded. A pricing test needs an outcome window that matches
  the shelf life, or it will favour clearance; the framework shows that rather than hiding it.
- A buy-side offer test randomised by seller: the surrogate index predicted +20.4 points in the
  120-day seller return rate after 30 days; the realised effect four months later was +17.4
  (+13.1 to +21.7).

**Engineering.** All injected bad rows quarantined, no good rows lost; 4.3 million engine prices
re-checked independently against the guardrails with 0 violations; buy quote for 25 items p95
29 ms; no buying into a full warehouse in any run.

## How it works

```
simulated market (hidden truth) ──► raw exports ──► contracts ──► warehouse (DuckDB) ──► market replay
                                                     quarantine                              │
                                                                                             ▼
 market price · demand & price response · seller acceptance · grading · seller value   (models)
                                                                                             │
                                                                                             ▼
 buy offers · buy/no-buy · sell prices · liquidation · shelf price · steering rules     (engine)
        │                       │                          │
     HTTP API          weekly steering report      experiments (assign, CUPED, mSPRT, surrogate)
                                                                                             │
 evaluation against the truth: backtests, ablations, scenarios, model accuracy, experiment validity ──► release gate
```

- **Market price** ([models/market_price.py](src/rpe/models/market_price.py)): median of the last 28
  days of the competitor lowest price, switching to 7 days when it has moved by more than 25%;
  a LightGBM cold-start model (category, format, language, age, list price, sales rank) with
  quantile models for the range; blended by how much feed there is.
- **Demand** ([models/demand.py](src/rpe/models/demand.py)): daily sale intensity of a copy =
  frailty x title rate (LightGBM with a complementary log-log objective) x copies^-d x price share
  (logistic per category). Price response, condition factors and d are fitted only on rows from
  randomised price tests ([ADR 0003](docs/adr/0003-price-response-from-randomised-tests.md)); a
  gamma frailty, estimated on out-of-fold predictions, makes a copy that has waited look slower.
- **Seller side**: acceptance (logistic on the offer against the competitor buy-back quote, from
  the randomised offer tests), grading (multinomial), seller value (does a better offer bring the
  seller back, and what is a returning seller worth).
- **Engine** ([engine/core.py](src/rpe/engine/core.py)): value-based decisions within guardrails
  ([ADR 0002](docs/adr/0002-value-based-pricing-within-guardrails.md)); an extra copy is valued as
  a queue behind the copies already held ([ADR 0004](docs/adr/0004-copies-share-buyers.md)); the
  shelf price ([ADR 0005](docs/adr/0005-warehouse-shelf-price.md)).
- **Experiments** ([experiments/](src/rpe/experiments)): deterministic assignment by title or
  seller, Welch, CUPED, delta method for ratio metrics, sample-ratio check, power, always-valid
  p-values, surrogate index; registry in [config/experiments.yaml](config/experiments.yaml).
- **Steering** ([steering/](src/rpe/steering)): KPI tree (revenue, margin, contribution, sell-through,
  offers kept, baskets sent, average paid, fill, aged stock), alerts (price war watch, fill, aged
  stock, margin per unit, acceptance), market watch by category, the AI roadmap from the ablations:
  `reports/steering.html` and `.md`.
- **API** ([api.py](src/rpe/api.py)): `POST /v1/buy/quote` (seller app), `POST /v1/sell/prices`
  (repricer), `GET /v1/titles/{ean}`, experiment assignment, the steering report, `/metrics`,
  API keys with roles. Every answer explains itself (market price and its source, sellability,
  copies held, expected profit, the bound and rules that applied).

## Run it

```bash
pip install -e ".[dev]"     # Python 3.11+
rpe run --workers 4         # history, contracts, warehouse, models, API benchmark, evaluation, steering, gate (~25 min)
pytest -q                   # unit and pipeline tests
RPE_API_KEYS='{"dev-key": "pricing"}' rpe serve
curl -s -H 'X-API-Key: dev-key' -H 'content-type: application/json' \
  -d '{"items": ["9781000011531", "4000001530090"]}' localhost:8000/v1/buy/quote
```

Steps can be run one by one: `rpe history`, `ingest`, `warehouse`, `train`, `bench-api`, `eval`,
`steering`, `gate`, `report`. `docker build -t rpe .` serves trained models from `/models` and the
warehouse from `/data` ([docs/RUNBOOK.md](docs/RUNBOOK.md)).

## Layout

```
config/            world (simulator only), costs, pricing (engine and steering rules), baseline rules,
                   experiments, steering alerts, release gate, data contracts
scenarios/         dev and held-out stress scenarios (FIRST_RUN.md records the first held-out run)
src/rpe/
  sim/             the market: world, simulator, history export
  contracts.py     contracts and quarantine;  warehouse.py + sql/  DuckDB marts
  market.py        the observed market view;  catalog.py  title features
  models/          market price, demand, acceptance, grading, seller value, history replay
  engine/          valuation and decisions, guardrails and steering rules
  baseline.py      today's rules
  experiments/     assignment, analysis, sequential tests, surrogate index
  steering/        weekly steering report
  service.py, api.py, metrics.py   the pricing service
  evaluation/      backtests, model accuracy, experiment validity, scenarios, report, gate
docs/              architecture, decisions, roadmap, runbook, scenarios
```

## Limits

- **A simulated market.** The results hold for this world, whose parameters were fixed in the first
  commit, before the engine existed. Real demand, competitors and sellers will differ; the first
  real step is the experiment programme in the roadmap, not a switch-over.
- **The headline values the stock held at the end** at its true market price and demand, the same
  way for both policies. The two-year run checks it in cash, on one replay.
- **One sales channel, one warehouse.** Marketplace-specific fees and buy-box dynamics, several
  warehouses and transfer costs are not modelled.
- **Short tests mislead.** The 13-week price test favoured the rules while the six-month and
  two-year backtests favour the engine; test windows must match the shelf life.
- **The long-term term in offers** costs value over six months unless a returning seller is worth
  about 10 EUR or more. That is a business decision, exposed as `long_term_weight`.
- Competitor buy-back quotes exist for about half the titles; acceptance elsewhere relies on a
  category ratio.

## What changed after the first held-out run

The release gate had a check "no overflow in any stress scenario". Two scenarios cut capacity below
the stock already on the shelves on day one, so no policy can meet it, and whether to liquidate
valuable stock or pay for external storage is an economic choice rather than a guardrail. The check
was replaced by "never buys while stock and parcels exceed the shelves" (blocking), with overflow
against the rules still reported. The engine was not changed after the held-out run; the first-run
numbers above are the only ones.
