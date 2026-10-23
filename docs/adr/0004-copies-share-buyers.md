# 0004: An extra copy of a title queues behind the copies we already hold

## Context

A first version valued each new copy as if it had its own buyers. It bought many copies of
popular titles and then had to liquidate them: buyers of a title take one copy each, so a tenth copy
adds no sales now; it sells only after the nine ahead of it.

## Decision

The demand model splits a title's demand over our copies (intensity ~ copies^-d, d estimated from
the tests, close to 1). When buying, the engine values the k-th copy as a queue: with title sales at
rate L it sells at a Gamma(k, L) time, so P(sold by W) = P(Poisson(LW) >= k) and its expected days in
stock follow in closed form. Copies on the way and items scanned earlier the same day count.

## Consequences

- Offers fall smoothly as stock of a title builds up; buy/no-buy becomes depth-aware.
- It is the single largest lever on the buy side; the tests check that more copies never raise an offer.
