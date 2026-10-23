# Scenarios

A scenario changes the simulated market during the forward period (the six months after the 2025
history, 2026-01-01 to 2026-07-01) to see how a pricing policy copes with something the history did
not show. Every policy meets the same scenario with the same sellers and the same random draws.

Scenarios are YAML files:

```yaml
name: fiction-price-war            # short id, used in reports
description: >                     # what happens in the market, in business terms
  A competitor dumps fiction stock: fiction market prices fall 25% over six weeks and stay there.
capacity_factor: 1.0               # optional: warehouse capacity multiplier from day 0 (0.7 = 30% fewer slots)
overrides:                         # optional list of market changes
  - what: market_price             # market_price | demand | sessions
    factor: 0.75                   # multiplier reached at the end of the ramp
    categories: [fiction]          # optional; all titles when omitted (ignored for sessions)
    start_day: 14                  # days after the start of the forward period
    ramp_days: 42                  # linear ramp from 1.0 to `factor` (1 = a step)
    end_day: 150                   # optional; back to normal after this day
```

What each override changes:

- `market_price`: the true market price of used copies of the matching titles. Competitor lowest
  prices in the feed and competitor buy-back quotes follow it (they are derived from it).
- `demand`: how many buyers want the matching titles (a demand shock, e.g. a TV adaptation, a
  curriculum change, a recession).
- `sessions`: how many new sellers open the app per day (marketing campaigns, a competitor closing,
  the spring clean-out).
- `capacity_factor`: the warehouse loses or gains shelf slots for the whole period.

Categories: fiction, nonfiction, children, textbooks, cookbooks, comics, cd, dvd, bluray, games.
Several overrides can be combined; their factors multiply.

`scenarios/dev/` holds the scenarios used while building the engine. `scenarios/heldout/` holds
scenarios written by someone who had not seen the engine or its results; each is run once and its
first-run result is recorded in `scenarios/heldout/FIRST_RUN.md` before anything is changed.
