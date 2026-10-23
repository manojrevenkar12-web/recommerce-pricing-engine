"""The company's view of the market, built only from what it observes: the competitor lowest-price
feed, competitor buy-back quotes and sales ranks. Shared by the rule-based policy and the engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpe.policy import MarketUpdate

WINDOW = 56  # days of feed kept per title
OWN_DECAY = float(np.exp(-1 / 60))  # own sales and listed days are counted with a 60-day memory


class MarketView:
    def __init__(self, n_titles: int):
        self.n = n_titles
        self.last_price = np.full(n_titles, np.nan)
        self.last_day = np.full(n_titles, -(10**6))
        self.offers = np.zeros(n_titles)
        self.quote = np.full(n_titles, np.nan)
        self.quote_day = np.full(n_titles, -(10**6))
        self.rank = np.full(n_titles, np.nan)
        self.ever_seen = np.zeros(n_titles, bool)
        self._log = np.full((WINDOW, n_titles), np.nan, dtype=np.float32)  # ring buffer of log feed prices
        self._ring_day = np.full(WINDOW, -(10**6))
        self.day = -1
        # our own sales and listed unit-days per title, exponentially decayed
        self.own_sales = np.zeros(n_titles)
        self.own_expo = np.zeros(n_titles)
        self.own_day = -1

    def observe(self, u: MarketUpdate) -> None:
        if u.day != self.day:
            slot = u.day % WINDOW
            self._log[slot] = np.nan
            self._ring_day[slot] = u.day
            self.day = u.day
        slot = u.day % WINDOW
        if len(u.feed_idx):
            self.last_price[u.feed_idx] = u.feed_price
            self.last_day[u.feed_idx] = u.day
            self.offers[u.feed_idx] = u.feed_offers
            self.ever_seen[u.feed_idx] = True
            self._log[slot, u.feed_idx] = np.log(u.feed_price)
        if len(u.quote_idx):
            self.quote[u.quote_idx] = u.quote
            self.quote_day[u.quote_idx] = u.day
        if len(u.rank_idx):
            self.rank[u.rank_idx] = u.rank
        if len(u.sold_idx):
            self._decay_to(u.day)
            np.add.at(self.own_sales, u.sold_idx, 1.0)

    def _decay_to(self, day: int) -> None:
        if self.own_day < 0:
            self.own_day = day
        elif day > self.own_day:
            f = OWN_DECAY ** (day - self.own_day)
            self.own_sales *= f
            self.own_expo *= f
            self.own_day = day

    def observe_stock(self, day: int, listed_per_title: np.ndarray) -> None:
        """Our listed copies per title today (once per day)."""
        if day == getattr(self, "_stock_day", None):
            return
        self._stock_day = day
        self._decay_to(day)
        self.own_expo += listed_per_title

    def recent_log_prices(self, days: int) -> np.ndarray:
        """(days x titles) log feed prices of the last `days` days, NaN where missing."""
        order = np.argsort(self._ring_day)[::-1]
        keep = [s for s in order if self._ring_day[s] > self.day - days]
        return self._log[keep]

    @classmethod
    def from_history(
        cls, n_titles: int, feed: pd.DataFrame, quotes: pd.DataFrame, ranks: pd.DataFrame, as_of: int, units: pd.DataFrame | None = None
    ) -> MarketView:
        """Rebuild the view as it stood at the end of `as_of` from the clean history tables (with a
        title index `idx` and a `day` column; `units` from the warehouse for our own sales)."""
        mv = cls(n_titles)
        if units is not None:
            first = int(feed.day.min())
            for d, sales, stock in own_history(n_titles, units, first, as_of):
                mv._decay_to(d)
                mv.own_sales += sales
                mv.own_expo += stock
            mv._stock_day = as_of
        f = feed[(feed.day > as_of - WINDOW) & (feed.day <= as_of)]
        for d, g in f.groupby("day", sort=True):
            mv.observe(MarketUpdate(int(d), g.idx.to_numpy(), g.price.to_numpy(), g.n_offers.to_numpy()))
        old = feed[feed.day <= as_of].sort_values("day").groupby("idx").tail(1)
        miss = ~mv.ever_seen[old.idx.to_numpy()]
        o = old[miss]
        mv.last_price[o.idx.to_numpy()] = o.price.to_numpy()
        mv.last_day[o.idx.to_numpy()] = o.day.to_numpy()
        mv.offers[o.idx.to_numpy()] = o.n_offers.to_numpy()
        mv.ever_seen[feed.idx.unique()] = True
        q = quotes[quotes.day <= as_of].sort_values("day").groupby("idx").tail(1)
        mv.quote[q.idx.to_numpy()] = q.quote.to_numpy()
        mv.quote_day[q.idx.to_numpy()] = q.day.to_numpy()
        r = ranks[ranks.day <= as_of].sort_values("day").groupby("idx").tail(1)
        mv.rank[r.idx.to_numpy()] = r["rank"].to_numpy()
        mv.day = as_of
        return mv


def own_history(n_titles: int, units: pd.DataFrame, first: int, last: int):
    """Per day from `first` to `last`: (day, our sales of the previous day, our listed copies),
    per title, from the warehouse units table."""
    days = last - first + 2
    listed = units.listed_day.to_numpy().astype(int)
    end = np.select(
        [units.sold_day.notna(), units.disposed_day.notna()], [units.sold_day.fillna(0) + 1, units.disposed_day.fillna(0)], last + 1
    ).astype(int)
    idx = units.idx.to_numpy().astype(int)
    diff = np.zeros((days + 1, n_titles), dtype=np.float32)
    a = np.clip(listed - first, 0, days)
    b = np.clip(end - first, 0, days)
    np.add.at(diff, (a, idx), 1)
    np.add.at(diff, (b, idx), -1)
    stock = np.cumsum(diff, axis=0)
    sold = units.sold_day.notna().to_numpy()
    sd = units.sold_day.to_numpy()[sold].astype(int) - first
    ok = (sd >= 0) & (sd < days)
    sales = np.zeros((days + 1, n_titles), dtype=np.float32)
    np.add.at(sales, (sd[ok] + 1, idx[sold][ok]), 1)  # seen the day after
    for k in range(days - 1):
        yield first + k, sales[k], stock[k]
