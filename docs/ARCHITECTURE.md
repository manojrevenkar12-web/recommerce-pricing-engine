# Architecture

```
 simulated market (hidden truth)          what the company sees                       decisions
 ───────────────────────────────          ─────────────────────                       ─────────
 titles: demand, market price,     ──►  data/raw: catalogue, competitor feed,  ──►  contracts ──► warehouse (DuckDB)
 price response, condition               sales ranks, buy-back quotes, offers,          │              │
 sellers: reservation, return            baskets, receipts, prices, sales,     quarantine            ▼
 warehouse: capacity                     disposals, warehouse levels                 replay of the market, weekly snapshots
        ▲                                                                                     │
        │                                                                                     ▼
        │                                                                  models: market price · demand & price response
        │                                                                  · seller acceptance · grading · seller value
        │                                                                                     │
        │                                                                                     ▼
        │     buy offers / sell prices / liquidations                  pricing engine: buy · sell · shelf price · guardrails
        └────────────────────────────────────────────────────────────  │            │                    │
                                                                        API        steering report      experiments
 evaluation (reads the truth): backtests, ablations, scenarios, model accuracy, experiment validity  ──►  release gate
```

## Packages

| package | does | reads the truth |
|---|---|---|
| `rpe.sim` | the market: world, day-by-day simulator, 2025 history export | yes |
| `rpe.contracts`, `rpe.warehouse` | contracts on every raw export, DuckDB marts (`src/rpe/sql`) | no |
| `rpe.market`, `rpe.catalog` | the observed market view (feed ring buffer, quotes, ranks, own sales), title features | no |
| `rpe.models` | market price, demand (rate, price response, copies, frailty), acceptance, grading, seller value; history replay | no |
| `rpe.engine` | valuation, buy/sell decisions, shelf price, steering rules and guardrails | no |
| `rpe.baseline` | today's rule-based pricing, the control everywhere | no |
| `rpe.experiments` | assignment, Welch/CUPED/delta method, SRM, mSPRT, power, surrogate index | no |
| `rpe.service`, `rpe.api`, `rpe.metrics` | the pricing service, HTTP API, Prometheus metrics | no |
| `rpe.steering` | weekly steering report for business-unit management | no |
| `rpe.evaluation` | backtests (parallel), model accuracy, experiment validity, scenarios, report, gate | yes |

## Decisions

- [0001](adr/0001-simulated-market-with-hidden-truth.md) Evaluate on a simulated market with a hidden truth
- [0002](adr/0002-value-based-pricing-within-guardrails.md) Price by expected value, within guardrails applied before optimisation
- [0003](adr/0003-price-response-from-randomised-tests.md) Estimate price response only from randomised tests
- [0004](adr/0004-copies-share-buyers.md) An extra copy queues behind the copies we hold
- [0005](adr/0005-warehouse-shelf-price.md) Balance pricing with warehouse capacity through a shelf price

## Train and serve the same way

Training replays the observed market day by day and takes weekly snapshots, so every model learns
from the features it will be given in production (no hindsight). The engine computes the same
features from the live market view; the API builds that view from the warehouse at start-up.

## Runtime

A six-month backtest of the engine takes about 80 s on one core (the rule-based policy 2 s);
`rpe eval` runs its backtests in parallel processes (`RPE_WORKERS`). A buy quote for 25 items takes
about 25 ms (p95 under 30 ms) in-process.
