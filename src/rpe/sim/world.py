"""The simulated catalogue and its hidden truth: demand, market price, price response, condition.

A `World` holds per-title arrays. Some are observable (the catalogue attributes a company gets from
its wholesale data feed) and some are hidden (true demand, true market price, true price response).
Only the simulator and the evaluator use the hidden ones.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from rpe import ean
from rpe.config import load

BOOKS = {"fiction", "nonfiction", "children", "textbooks", "cookbooks", "comics"}
PREFIX = {"cd": "40", "dvd": "41", "bluray": "42", "games": "43"}


def _d(x: date | str) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def bumps(spec: list[list[float]], doy: int | np.ndarray) -> np.ndarray | float:
    """1 + sum of gaussian bumps on the circular day of year."""
    out = np.ones_like(np.asarray(doy, dtype=float))
    for centre, width, height in spec:
        d = np.abs(np.asarray(doy, dtype=float) - centre)
        d = np.minimum(d, 365 - d)
        out = out + height * np.exp(-0.5 * (d / width) ** 2)
    return out


@dataclass
class World:
    cfg: dict[str, Any]
    start: date
    categories: list[str]
    # observable catalogue
    ean: np.ndarray
    category: np.ndarray  # int index into categories
    fmt: np.ndarray
    language: np.ndarray
    pub_year: np.ndarray
    new_price: np.ndarray  # NaN when the feed has no list price
    release_day: np.ndarray
    # hidden truth
    lam0: np.ndarray  # demand per day for our listings when priced at share 1
    m0: np.ndarray  # market price, very good condition, at history start
    beta: np.ndarray
    alpha: np.ndarray
    drift: np.ndarray
    kappa: np.ndarray  # competitor buy-back quote / market price
    covered: np.ndarray  # has competitor offers in the price feed
    quoted: np.ndarray  # has a scraped competitor buy-back quote
    edition_day: np.ndarray  # day a new edition replaces this one, or -1
    grade_shift: np.ndarray
    supply_w: np.ndarray
    season_idx: np.ndarray
    season_names: list[str]
    overrides: list[dict[str, Any]] = field(default_factory=list)
    capacity: int = 0

    # -- calendar -----------------------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.ean)

    def date(self, day: int) -> date:
        return self.start + timedelta(days=int(day))

    def day_of(self, d: date | str) -> int:
        return (_d(d) - self.start).days

    @property
    def history_start(self) -> int:
        return self.day_of(self.cfg["calendar"]["history_start"])

    @property
    def history_end(self) -> int:
        return self.day_of(self.cfg["calendar"]["history_end"])

    def doy(self, day: int) -> int:
        return self.date(day).timetuple().tm_yday

    # -- truth over time ----------------------------------------------------------------------
    def _override_factor(self, day: int, what: str) -> np.ndarray:
        f = np.ones(self.n)
        for o in self.overrides:
            if o.get("what") != what:
                continue
            start, ramp = o["start_day"], max(1, o.get("ramp_days", 1))
            if day < start or ("end_day" in o and day > o["end_day"]):
                continue
            k = min(1.0, (day - start + 1) / ramp)
            mask = np.ones(self.n, bool)
            if "categories" in o:
                mask &= np.isin(self.category, [self.categories.index(c) for c in o["categories"]])
            f[mask] *= 1 + k * (o["factor"] - 1)
        return f

    def season_factor(self, day: int) -> np.ndarray:
        doy = self.doy(day)
        curves = np.array([bumps(self.cfg["season"][s], doy) for s in self.season_names], dtype=float)
        return curves[self.season_idx]

    def lam(self, day: int) -> np.ndarray:
        c = self.cfg["catalogue"]
        released = day >= self.release_day
        ed = np.where((self.edition_day >= 0) & (day >= self.edition_day), c["edition_demand_factor"], 1.0)
        # demand ramps up over the first weeks after release
        since = np.clip((day - self.release_day) / 30.0, 0, 1)
        lam = self.lam0 * self.season_factor(day) * ed * np.where(released, 0.3 + 0.7 * since, 0.0)
        return lam * self._override_factor(day, "demand")

    def m(self, day: int) -> np.ndarray:
        c = self.cfg["catalogue"]
        years = (day - self.history_start) / 365.0
        ed = np.where((self.edition_day >= 0) & (day >= self.edition_day), c["edition_price_factor"], 1.0)
        m = self.m0 * np.exp(self.drift * years) * ed * self._override_factor(day, "market_price")
        return np.maximum(m, c["min_market_price"])

    def quote(self, day: int) -> np.ndarray:
        return self.m(day) * self.kappa

    def supply_weights(self, day: int) -> np.ndarray:
        age = day - self.release_day
        w = self.supply_w * np.clip(age / 90.0, 0, 1)
        return w / w.sum()

    def share(self, rel_log_price: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Share of a title's demand that our listing gets at log(price / market price for its condition)."""
        return sigmoid(-self.beta[idx] * (rel_log_price - self.alpha[idx]))

    def condition_price_factor(self) -> np.ndarray:
        q = self.cfg["condition"]["price_factor"]
        return np.array([q[g] for g in self.cfg["condition"]["grades"]])

    def grade(self, idx: np.ndarray, day: int, seller_quality: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Condition on arrival: 0..3 = like_new..acceptable, 4 = reject."""
        cfg = self.cfg["condition"]
        years = np.maximum(0, self.date(day).year - self.pub_year[idx])
        latent = rng.standard_normal(len(idx)) + self.grade_shift[idx] + cfg["age_shift_per_year"] * years + seller_quality
        return np.searchsorted(np.array(cfg["cutpoints"]), latent)

    # -- export -------------------------------------------------------------------------------
    def catalogue_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ean": self.ean,
                "category": np.array(self.categories)[self.category],
                "format": self.fmt,
                "language": self.language,
                "pub_year": self.pub_year,
                "release_date": [self.date(d).isoformat() for d in self.release_day],
                "new_price": np.round(self.new_price, 2),
            }
        )

    def truth_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ean": self.ean,
                "lam0": self.lam0,
                "m0": self.m0,
                "beta": self.beta,
                "alpha": self.alpha,
                "drift": self.drift,
                "kappa": self.kappa,
                "covered": self.covered,
                "quoted": self.quoted,
                "edition_day": self.edition_day,
                "grade_shift": self.grade_shift,
            }
        )

    def with_scenario(self, scenario: dict[str, Any]) -> World:
        """A copy with a scenario's overrides (see docs/SCENARIOS.md). Days are relative to the
        start of the forward period."""
        w = copy.copy(self)
        w.overrides = []
        f0 = self.history_end + 1
        for o in scenario.get("overrides", []):
            o = dict(o)
            o["start_day"] = f0 + int(o.get("start_day", 0))
            if "end_day" in o:
                o["end_day"] = f0 + int(o["end_day"])
            w.overrides.append(o)
        w.capacity = int(self.capacity * scenario.get("capacity_factor", 1.0))
        return w


def build(cfg: dict[str, Any] | None = None, seed: int | None = None) -> World:
    cfg = cfg or load("world")
    rng = np.random.default_rng(cfg["seed"] if seed is None else seed)
    c = cfg["catalogue"]
    cats = list(c["categories"])
    p = np.array([c["categories"][k]["share"] for k in cats])
    n = c["titles"]
    cat = rng.choice(len(cats), size=n, p=p / p.sum())
    names = np.array(cats)[cat]
    P = {k: np.array([c["categories"][x][k] for x in names], dtype=float) for k in ("new_price", "value_ratio", "beta", "alpha", "pop_mu", "drift", "grade_shift", "kappa")}

    is_book = np.isin(names, list(BOOKS))
    fmt = np.empty(n, dtype=object)
    for kind, mask in (("books", is_book), ("media", ~is_book)):
        opts = c["formats"][kind]
        fmt[mask] = rng.choice(list(opts), size=mask.sum(), p=np.array(list(opts.values())) / sum(opts.values()))
    language = np.where(rng.random(n) < c["english_share"], "en", "de")

    # publication: skewed to recent years; some titles release during history or the forward period
    start = _d(cfg["calendar"]["burn_in_start"])
    h0 = (_d(cfg["calendar"]["history_start"]) - start).days
    h1 = (_d(cfg["calendar"]["history_end"]) - start).days
    u = rng.random(n)
    back = -np.minimum(rng.exponential(365 * 7, n), 365 * 30).astype(int)
    release_day = np.where(
        u < 0.04,
        rng.integers(h0, h1 + 1, n),  # released during history
        np.where(u < 0.07, rng.integers(h1 + 1, h1 + 1 + cfg["calendar"]["forward_days"], n), back),
    )
    pub_year = np.array([(start + timedelta(days=int(d))).year for d in release_day])

    new_price = P["new_price"] * np.exp(rng.normal(0, 0.30, n))
    new_price *= np.array([c["format_value"][f] for f in fmt])
    new_price = np.round(np.maximum(new_price, 4.99), 0) - 0.01

    z = rng.standard_normal(n)
    lam0 = np.exp(P["pop_mu"] + c["demand_sigma"] * z)
    years = np.maximum(0, 2025 - pub_year)
    age_el = np.where(names == "games", c["games_age_elasticity"], c["title_age_elasticity"])
    m0 = (
        new_price
        * P["value_ratio"]
        * (1 + years) ** age_el
        * np.where(language == "en", c["english_value"], 1.0)
        * np.clip(lam0 / 0.01, 0.05, 50) ** c["demand_value_elasticity"]
        * np.exp(rng.normal(0, c["value_sigma"], n))
    )
    m0 = np.maximum(m0, c["min_market_price"])

    f = cfg["market_feed"]
    covered = rng.random(n) < sigmoid(f["coverage_intercept"] + f["coverage_slope"] * np.log(lam0 / 0.01))
    quoted = rng.random(n) < f["quote_coverage"]
    kappa = P["kappa"] * np.exp(rng.normal(0, 0.25, n))

    total_days = h1 + 1 + cfg["calendar"]["forward_days"]
    edition_day = np.full(n, -1)
    tb = names == "textbooks"
    hit = tb & (rng.random(n) < c["edition_change_per_year"] * total_days / 365)
    edition_day[hit] = rng.integers(0, total_days, hit.sum())

    season_names = list(cfg["season"])
    season_idx = np.array([season_names.index(c["categories"][x]["season"]) for x in names])

    codes = np.empty(n, dtype=object)
    serial = rng.permutation(10**6)[:n]
    for i in range(n):
        codes[i] = ean.make("978" + ("3" if language[i] == "de" else "1"), serial[i]) if is_book[i] else ean.make(PREFIX[names[i]], serial[i])

    np_obs = new_price.copy()
    np_obs[rng.random(n) < c["missing_new_price"]] = np.nan

    order = np.argsort(codes)
    w = World(
        cfg=cfg,
        start=start,
        categories=cats,
        ean=codes[order].astype(str),
        category=cat[order],
        fmt=fmt[order].astype(str),
        language=language[order],
        pub_year=pub_year[order],
        new_price=np_obs[order],
        release_day=release_day[order],
        lam0=lam0[order],
        m0=m0[order],
        beta=P["beta"][order],
        alpha=P["alpha"][order],
        drift=P["drift"][order],
        kappa=kappa[order],
        covered=covered[order],
        quoted=quoted[order],
        edition_day=edition_day[order],
        grade_shift=P["grade_shift"][order],
        supply_w=(lam0**cfg["sellers"]["supply_demand_power"])[order],
        season_idx=season_idx[order],
        season_names=season_names,
        capacity=cfg["warehouse"]["capacity_units"],
    )
    w._true_new_price = new_price[order]  # type: ignore[attr-defined]
    return w
