# 0005: Balance pricing with warehouse capacity through a shelf price

## Context

The rules stop all buying at 97% fill, a cliff that loses good items and keeps slow ones.

## Decision

A controller sets a price per shelf slot per day from projected fill (stock plus parcels on the
way, plus the smoothed trend, against a target). The price is added to holding cost on both sides:
a fuller warehouse buys only copies that will sell fast and prices and liquidates slow stock
sooner. Liquidation is capped per week (no fire sales) and an emergency stop remains as a backstop.

## Consequences

- Capacity is a cost the optimisation sees, not a rule outside it. Where a scenario cuts capacity
  below the stock already held, the engine stops buying until it is back under capacity; its
  overflow was a third lower than the rules' in the warehouse-squeeze scenario and about the same in
  the capacity cut (477 vs 419 unit-days). Clearing overflow faster is a liquidation-policy choice.
- The target fill and the controller step are business settings in `config/pricing.yaml`.
