"""Replays the observed market of the history day by day, as the pricing system saw it, and takes
weekly snapshots. Training uses these snapshots so a model learns from exactly what it will be
given in production, not from hindsight."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rpe.market import MarketView, own_history
from rpe.models.market_price import Snapshot, snapshot
from rpe.policy import MarketUpdate


class _ByDay:
    def __init__(self, df: pd.DataFrame, cols: list[str]):
        df = df.sort_values("day", kind="stable")
        self.day = df.day.to_numpy()
        self.cols = {c: df[c].to_numpy() for c in cols}

    def get(self, d: int) -> dict[str, np.ndarray]:
        a, b = np.searchsorted(self.day, d, "left"), np.searchsorted(self.day, d, "right")
        return {k: v[a:b] for k, v in self.cols.items()}


def replay(
    n_titles: int,
    feed: pd.DataFrame,
    quotes: pd.DataFrame,
    ranks: pd.DataFrame,
    first: int,
    last: int,
    snapshot_days: set[int],
    units: pd.DataFrame | None = None,
) -> tuple[dict[int, Snapshot], MarketView]:
    mv = MarketView(n_titles)
    own = own_history(n_titles, units, first, last) if units is not None else None
    F, Q, R = _ByDay(feed, ["idx", "price", "n_offers"]), _ByDay(quotes, ["idx", "quote"]), _ByDay(ranks, ["idx", "rank"])
    snaps: dict[int, Snapshot] = {}
    for d in range(first, last + 1):
        f, q, r = F.get(d), Q.get(d), R.get(d)
        mv.observe(MarketUpdate(d, f["idx"], f["price"], f["n_offers"], q["idx"], q["quote"], r["idx"], r["rank"]))
        if own is not None:
            _, sales, stock = next(own)
            mv._decay_to(d)
            mv.own_sales += sales
            mv.observe_stock(d, stock)
        if d in snapshot_days:
            snaps[d] = snapshot(mv, d)
    return snaps, mv
