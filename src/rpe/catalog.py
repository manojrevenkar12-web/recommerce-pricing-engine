"""The title catalogue as arrays indexed by title (`idx`), and the features every model shares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rpe.calendar import to_date

CATEGORIES = ["bluray", "cd", "children", "comics", "cookbooks", "dvd", "fiction", "games", "nonfiction", "textbooks"]
FORMATS = ["hardcover", "paperback", "special_edition", "standard"]
BOOKS = {"fiction", "nonfiction", "children", "textbooks", "cookbooks", "comics"}


@dataclass
class Catalog:
    ean: np.ndarray
    category: np.ndarray  # code into CATEGORIES
    fmt: np.ndarray
    english: np.ndarray
    pub_year: np.ndarray
    release_day: np.ndarray
    new_price: np.ndarray

    @property
    def n(self) -> int:
        return len(self.ean)

    @classmethod
    def from_frame(cls, t: pd.DataFrame) -> Catalog:
        t = t.sort_values("idx")
        assert (t.idx.to_numpy() == np.arange(len(t))).all(), "dim_title must be dense"
        return cls(
            ean=t.ean.to_numpy().astype(str),
            category=t.category.map({c: i for i, c in enumerate(CATEGORIES)}).to_numpy().astype(int),
            fmt=t.format.map({c: i for i, c in enumerate(FORMATS)}).to_numpy().astype(int),
            english=(t.language == "en").to_numpy().astype(int),
            pub_year=t.pub_year.to_numpy().astype(int),
            release_day=t.release_day.to_numpy().astype(int),
            new_price=t.new_price.to_numpy().astype(float),
        )

    def index_of(self, eans: list[str] | np.ndarray) -> np.ndarray:
        """Title index per EAN, -1 when unknown."""
        pos = np.searchsorted(self.ean, eans)
        pos = np.clip(pos, 0, self.n - 1)
        return np.where(self.ean[pos] == np.asarray(eans), pos, -1)


TITLE_FEATURES = [
    "category",
    "fmt",
    "english",
    "title_age",
    "days_since_release",
    "log_new_price",
    "log_rank",
    "log_offers",
    "has_feed",
    "log_market",
    "market_obs",
    "doy_sin",
    "doy_cos",
    "own_rate",
    "own_listed_days",
]


def title_features(
    cat: Catalog, mv: Any, day: int, log_market: np.ndarray, market_obs: np.ndarray, idx: np.ndarray | None = None
) -> pd.DataFrame:
    """Features of titles as of `day`, from the catalogue and the observed market only (`mv` is a
    MarketView or a Snapshot)."""
    idx = np.arange(cat.n) if idx is None else idx
    year = to_date(day).year
    doy = to_date(day).timetuple().tm_yday
    rank = mv.rank[idx]
    return pd.DataFrame(
        {
            "category": cat.category[idx],
            "fmt": cat.fmt[idx],
            "english": cat.english[idx],
            "title_age": np.maximum(0, year - cat.pub_year[idx]),
            "days_since_release": np.clip(day - cat.release_day[idx], -30, 3650),
            "log_new_price": np.log(cat.new_price[idx]),
            "log_rank": np.log(rank),
            "log_offers": np.log1p(mv.offers[idx]),
            "has_feed": mv.ever_seen[idx].astype(int),
            "log_market": log_market[idx],
            "market_obs": market_obs[idx],
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            # our own sales per listed day of this title (shrunk toward a slow seller), and how much evidence there is
            "own_rate": np.log((mv.own_sales[idx] + 0.25) / (mv.own_expo[idx] + 60.0)),
            "own_listed_days": np.log1p(mv.own_expo[idx]),
        }
    )
