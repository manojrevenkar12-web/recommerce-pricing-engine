# 0002: Price by expected value, within guardrails applied before optimisation

## Context

The rule-based pricing lists one cent below the lowest competitor and marks down by age; it buys at
a share of the competitor price by sales-rank bucket. Neither looks at how fast a copy will sell,
how many copies we already hold, what the warehouse costs or what a seller will accept.

## Decision

Both sides maximise expected value from the models:

- Selling: for each copy, the price that maximises expected net proceeds over its remaining shelf
  life (sale probability and expected days listed from the demand model, holding cost plus the
  warehouse shelf price, liquidation at the end). Liquidate when keeping is worth less than bulk.
- Buying: for each scanned item, the offer that maximises P(seller keeps it) x (expected value of
  the copy that arrives, across grades and rejects, minus handling and parcel cost, minus the
  offer) plus the long-term value of a more generous basket; no offer below a minimum profit or
  for titles unlikely to sell.

Guardrails (minimum price, share of list price, steering bands by category/format/condition,
maximum weekly change, maximum offer share, emergency stop) define the candidate set before
optimisation, in a fixed priority order; an independent checker re-verifies every price in
evaluation. Steering rules live in `config/pricing.yaml`, owned by the pricing lead.

## Consequences

- Markdowns, liquidation and buy/no-buy follow from one economic objective instead of separate tables.
- Business owners change positioning in configuration, and every decision says which rules bound it.
- Decisions are only as good as the models; the evaluation measures both.
