# 0001: Evaluate on a simulated market with a hidden truth

## Context

Pricing decisions change what is bought, what sells and which sellers come back. A model's accuracy
on historical data says little about the profit of a policy, and live A/B tests are slow and costly.
The project also cannot use a real company's data.

## Decision

A simulator plays the market: titles with true demand, true market prices and true price response;
sellers with reservation prices who come back if they were treated well; condition grading; a
warehouse with a capacity. The current rule-based pricing runs it for 2025 (with the randomised
offer and price tests Pricing ran), and only what the company would see is exported: catalogue,
competitor price feed, sales ranks, competitor buy-back quotes, offers, baskets, receipts, prices,
sales, disposals and warehouse levels. The exports carry injected bad rows.

Models learn only from those exports (through the contracts and the warehouse). Policies are scored
by running them forward from the same end-of-2025 state with the same sellers and market (common
random numbers), against the truth. The world's parameters were committed before the engine existed.

## Consequences

- Policy value, not just model fit, is measurable, with intervals over market replays.
- Experiments can be validated against the true effect, because the simulator can run both worlds.
- Results hold for this world. The README says so; the held-out scenarios probe situations the
  history did not contain.
- `src/rpe/sim` and `src/rpe/evaluation` are the only readers of the truth; a test enforces it.
