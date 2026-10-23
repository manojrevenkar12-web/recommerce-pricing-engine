"""Guardrails and steering rules: the bounds every engine decision is optimised within.

Bounds are applied before optimisation (the engine only considers prices and offers that satisfy
them), so a decision is never "optimised then clipped". Each decision records which bounds bind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rpe.catalog import CATEGORIES, FORMATS
from rpe.policy import GRADES


@dataclass
class SellBounds:
    min_index: np.ndarray
    max_index: np.ndarray
    rules: list[list[str]]


@dataclass
class BuyBounds:
    max_offer_share: np.ndarray
    rules: list[list[str]]


def _matches(rule: dict[str, Any], category: np.ndarray, fmt: np.ndarray, grade: np.ndarray | None) -> np.ndarray:
    m = np.ones(len(category), bool)
    for k, v in rule.get("match", {}).items():
        vals = v if isinstance(v, list) else [v]
        if k == "category":
            m &= np.isin(category, [CATEGORIES.index(x) for x in vals])
        elif k == "format":
            m &= np.isin(fmt, [FORMATS.index(x) for x in vals])
        elif k == "grade":
            if grade is None:
                m &= False
            else:
                m &= np.isin(grade, [GRADES.index(x) for x in vals])
        else:
            raise ValueError(f"steering rule {rule.get('name')}: unknown match key {k}")
    return m


def validate(cfg: dict[str, Any]) -> list[str]:
    """Problems in a pricing config, for CI and for the API at start-up."""
    errs = []
    s, b = cfg["sell"], cfg["buy"]
    if not 0 < s["min_index"] < s["max_index"]:
        errs.append("sell.min_index must be positive and below sell.max_index")
    if not 0 < s["max_weekly_change"] < 1:
        errs.append("sell.max_weekly_change must be in (0, 1)")
    if not 0 < b["max_offer_share"] < 1:
        errs.append("buy.max_offer_share must be in (0, 1)")
    for r in cfg.get("steering", []):
        try:
            _matches(r, np.zeros(1, int), np.zeros(1, int), np.zeros(1, int))
        except (ValueError, KeyError) as e:
            errs.append(str(e))
        for side, keys in (("sell", {"min_index", "max_index"}), ("buy", {"max_offer_share"})):
            extra = set(r.get(side, {})) - keys
            if extra:
                errs.append(f"steering rule {r.get('name')}: unknown {side} settings {sorted(extra)}")
        si = r.get("sell", {})
        if si.get("max_index", 9) <= si.get("min_index", 0):
            errs.append(f"steering rule {r.get('name')}: max_index below min_index")
    return errs


def sell_bounds(cfg: dict[str, Any], category: np.ndarray, fmt: np.ndarray, grade: np.ndarray) -> SellBounds:
    n = len(category)
    lo = np.full(n, float(cfg["sell"]["min_index"]))
    hi = np.full(n, float(cfg["sell"]["max_index"]))
    names: list[list[str]] = [[] for _ in range(n)]
    for r in cfg.get("steering", []):
        if "sell" not in r:
            continue
        m = _matches(r, category, fmt, grade)
        if not m.any():
            continue
        if "min_index" in r["sell"]:
            lo = np.where(m, np.maximum(lo, r["sell"]["min_index"]), lo)
        if "max_index" in r["sell"]:
            hi = np.where(m, np.minimum(hi, r["sell"]["max_index"]), hi)
        for i in np.flatnonzero(m):
            names[i].append(r["name"])
    return SellBounds(lo, np.maximum(hi, lo), names)


def buy_bounds(cfg: dict[str, Any], category: np.ndarray, fmt: np.ndarray) -> BuyBounds:
    n = len(category)
    share = np.full(n, float(cfg["buy"]["max_offer_share"]))
    names: list[list[str]] = [[] for _ in range(n)]
    for r in cfg.get("steering", []):
        if "buy" not in r:
            continue
        m = _matches(r, category, fmt, None)
        if "max_offer_share" in r["buy"]:
            share = np.where(m, np.minimum(share, r["buy"]["max_offer_share"]), share)
        for i in np.flatnonzero(m):
            names[i].append(r["name"])
    return BuyBounds(share, names)


def price_interval(
    cfg: dict[str, Any], costs: dict[str, Any], ref: np.ndarray, old: np.ndarray, new_price: np.ndarray, bounds: SellBounds
) -> tuple[np.ndarray, np.ndarray]:
    """The allowed price range per unit. Bounds are intersected in priority order and a lower-priority
    bound gives way when it cannot be met together with the ones above it:
    minimum price > share of the list price > steering index band > maximum weekly change."""
    lo = np.full(len(ref), float(costs["sell"]["min_price"]))
    cap = cfg["sell"]["max_share_of_new_price"] * new_price
    hi = np.where(np.isfinite(cap), np.maximum(cap, lo), np.inf)
    lo, hi = _intersect(lo, hi, bounds.min_index * ref, bounds.max_index * ref)
    moved = np.isfinite(old) & (old > 0)
    k = cfg["sell"]["max_weekly_change"]
    wlo = np.where(moved, old * (1 - k), -np.inf)
    whi = np.where(moved, old * (1 + k), np.inf)
    lo, hi = _intersect(lo, hi, wlo, whi)
    return lo, np.where(np.isfinite(hi), hi, lo * 50)


def _intersect(lo: np.ndarray, hi: np.ndarray, blo: np.ndarray, bhi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[lo, hi] narrowed to [blo, bhi]; where they do not overlap, the point of [lo, hi] nearest to it."""
    nlo, nhi = np.maximum(lo, blo), np.minimum(hi, bhi)
    empty = nlo > nhi
    near = np.clip(np.where(bhi < lo, lo, hi), lo, hi)  # [blo, bhi] entirely below lo -> lo, above hi -> hi
    return np.where(empty, near, nlo), np.where(empty, near, nhi)


def sell_violations(
    cfg: dict[str, Any],
    costs: dict[str, Any],
    price: np.ndarray,
    old: np.ndarray,
    ref: np.ndarray,
    new_price: np.ndarray,
    bounds: SellBounds,
) -> dict[str, int]:
    """Independent re-check of priced units against each guardrail (used by the release gate). A
    lower-priority bound only counts where it could have been met together with the ones above it."""
    tol = 0.011
    min_price = float(costs["sell"]["min_price"])
    cap = np.where(np.isfinite(new_price), cfg["sell"]["max_share_of_new_price"] * new_price, np.inf)
    out = {"below_min_price": int((price < min_price - tol).sum())}
    cap_ok = cap >= min_price
    out["above_list_price_share"] = int((cap_ok & (price > cap + tol)).sum())
    band_lo, band_hi = bounds.min_index * ref, bounds.max_index * ref
    band_ok = (band_hi >= min_price) & (band_lo <= np.where(cap_ok, cap, np.inf))
    out["outside_steering_band"] = int((band_ok & ((price < band_lo - tol) | (price > band_hi + tol))).sum())
    k = cfg["sell"]["max_weekly_change"]
    moved = np.isfinite(old) & (old > 0)
    lo, hi = price_interval(cfg, costs, ref, np.full(len(ref), np.nan), new_price, bounds)
    week_ok = moved & (old * (1 + k) >= lo) & (old * (1 - k) <= hi)
    out["weekly_change_too_large"] = int((week_ok & (np.abs(price - np.where(moved, old, 0)) > k * np.where(moved, old, 0) + tol)).sum())
    return out
