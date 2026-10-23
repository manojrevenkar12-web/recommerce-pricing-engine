# Books & Media pricing: weekly steering, week ending 2025-12-31

## Where we stand (last 4 weeks vs the 4 before)

| | last 4 weeks | previous 4 | change |
|---|---|---|---|
| Revenue | 47,879 | 47,114 | +1.6% |
| Gross margin (revenue - purchase cost of units sold) | 37,970 | 37,313 | +1.8% |
| Contribution (after fees, handling, parcels, holding) | 13,825 | 14,418 | -4.1% |
| Units sold | 7,870 | 7,872 | -0.0% |
| Units received | 9,874 | 8,726 | +13.2% |
| Average sale price | 6.08 | 5.99 | |
| Average paid per unit | 1.04 | 1.09 | |
| Offers kept by sellers | 51.1% | 50.9% | |
| Baskets sent | 44.1% | 41.4% | |
| Scanned items we offer for | 68.7% | 69.1% | |

Stock: 26,954 units, warehouse fill 75%; 28% of stock listed for more than 180 days.

## Alerts

- **Old stock** (28% of stock listed for more than 180 days). Review liquidation of copies whose expected value is below bulk value.

## Market watch: competitor prices over 4 weeks

| category | change | titles |
|---|---|---|
| bluray | -0.8% | 273 |
| cd | +0.4% | 812 |
| children | +0.0% | 935 |
| comics | +0.7% | 421 |
| cookbooks | +0.5% | 529 |
| dvd | +0.3% | 770 |
| fiction | -0.4% | 2605 |
| games | -1.0% | 377 |
| nonfiction | -0.5% | 1531 |
| textbooks | -1.9% | 702 |

## The pricing engine against today's rules

Backtest over six months on 5 market replays: value (cash plus the value of stock at the end) +37.0% (95% interval +35.8% to +38.3%), cash in the period -23.7%; sellers who will come back +75.4%.

## AI roadmap, ranked by measured value (six months, EUR)

| use case | value | 95% interval |
|---|---|---|
| Warehouse-aware buying and selling (shelf price) | 30,652 | 28,984 to 32,320 |
| Buy offers and buy/no-buy (sellability, acceptance, copies in stock) | 26,284 | 25,848 to 26,720 |
| Market price definition (robust feed price, cold start for new titles) | 19,531 | 18,395 to 20,667 |
| Dynamic sell prices and liquidation (demand and price response) | 18,306 | 17,770 to 18,842 |
| Long-term seller value in offers | -5,312 | -5,482 to -5,142 |

## Experiments

- `sell-engine-2026-01` (by title): Engine sell prices earn more per title than the rule-based repricer on the same stock.
- `buy-engine-2026-01` (by seller): Engine offers raise expected profit per scanned item without hurting seller return rates.
