# 0003: Estimate price response only from randomised tests

## Context

In history, high prices go with titles nobody wants (the repricer follows the market, markdowns
follow slow sales). Fitting price response on all of history confuses the two and makes demand look
far less price sensitive than it is, or the reverse.

## Decision

The demand model is fitted in two alternating parts: a gradient-boosted rate of the title (no price
input) with the price response as an offset, and the price response itself (logistic share per
category, condition price factors, how copies share buyers) by maximum likelihood on the rows
where the price was moved by a randomised test. The same applies to sellers: offer tests identify
how acceptance moves with the offer.

## Consequences

- The recovered price response is close to the truth (see EVALUATION.md); it would not be otherwise.
- Pricing keeps running a small randomised test programme (20% of titles, 4-week cells), at a
  measurable cost, because the engine's quality depends on it. That is a standing roadmap item.
