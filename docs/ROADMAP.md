# Pricing & AI roadmap

How the work would be prioritised with the Pricing Engineering and AI Engineering teams: by measured
business impact, with the risks and the decisions management has to take. The values come from the
ablations in [reports/EVALUATION.md](../reports/EVALUATION.md): the engine with and without each
part, six months, same market replays.

## Where the value is

| # | Use case | Value, six months | Why it pays | Depends on |
|---|---|---|---|---|
| 1 | Warehouse-aware buying and selling (shelf price) | +30.7k EUR | the rules' 97% stop is a cliff: good items are lost and slow ones kept; a price per slot-day makes every decision feel the space | daily stock and parcel counts from Logistics IT |
| 2 | Buy offers and buy/no-buy | +26.3k EUR | pays more for what sells, less for duplicates and slow titles; baskets sent 42% to 57% | offer tests (randomised share of offers), grading data |
| 3 | Market price definition | +19.5k EUR | one dumping offer moves the rules' price; cold-start titles are priced at 40% of list, error 0.66 vs 0.30 | competitor feed, catalogue attributes |
| 4 | Dynamic sell prices and liquidation | +18.3k EUR | prices where demand allows, liquidates what is not worth its slot | price tests (20% of titles, 4-week cells) |
| 5 | Long-term seller value in offers | -5.3k EUR (+0.3k counting returning sellers at 10 EUR) | more sellers come back (+75% with the full engine), at a cost now | management decision below |

## Sequencing

1. **Run the test programme first.** Everything above depends on price response measured in
   randomised tests (ADR 0003). Keep 20% of titles in 4-week price cells and 15% of offers in offer
   tests; budget the cost explicitly.
2. **Market price, then buy/no-buy.** Lowest risk and needed by everything else: deploy behind the
   current rules (shadow mode), compare decisions daily, then switch category by category.
3. **Buy offers by A/B test, randomised by seller**, primary metric expected profit per scanned item,
   guardrails: baskets sent and the surrogate return rate; decision with always-valid p-values.
4. **Sell prices by A/B test, randomised by title**, with an outcome window of at least six months
   or a fair valuation of what is left: the 13-week version of this test favours clearance and
   showed the engine losing 0.4 to 0.7 EUR per title while the six-month backtest shows it winning.
5. **Shelf price last**: it couples both sides and cannot be A/B tested by unit (one warehouse);
   introduce it with a switchback design (alternating weeks) or a staged rollout with a holdout period.

## Decisions for business-unit management

- **Long-term weight in offers.** At 1.0 the engine trades about 5k EUR of six-month value for
  seller loyalty; worth it if a returning seller is worth 10 EUR or more. Marketing's cost per new
  seller who sends a parcel (2.17 EUR today, 1.54 with the engine) is the reference.
- **Cash now vs value later.** The engine spends more on better stock in its first months (cash
  -24% in the first half-year, +46% in the second year in the two-year run). Finance needs to plan
  for the working capital.
- **Liquidation policy.** At most 3% of stock per week, least valuable first; a higher cap clears a
  full warehouse faster (relevant in the warehouse-squeeze scenario) at bulk prices.
- **Positioning rules.** Steering bounds by category, format and condition (`config/pricing.yaml`)
  are owned by Pricing; each one costs value where it binds, and the API reports where it does.

## Risks and what watches them

| risk | watch |
|---|---|
| competitor feed outage: prices fall back to the cold-start model | share of titles priced from the feed, alert below 60% |
| a price war in a category | steering report market watch (alert at -10% in 4 weeks) |
| model drift after a season change | weekly retrain, the release gate on every model, model age metric |
| the test programme quietly stops | share of randomised offers and prices in the warehouse, checked by the pipeline tests |
| short-window tests leading to clearance pricing | outcome windows and valuation rules fixed in the experiment registry |
