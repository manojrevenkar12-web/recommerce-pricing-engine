"""The market simulator: plays sellers, buyers, competitors and the warehouse day by day against a
pricing policy, and keeps the books.

Random streams are keyed by (seed, purpose, day or seller visit), so two policies run with the same
seed meet the same sellers with the same items and the same market: differences in the result come
from the policies, not from luck (common random numbers).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpe.config import costs as load_costs
from rpe.policy import REJECT, Items, MarketUpdate, Policy, Units, Warehouse
from rpe.sim.world import World, bumps, sigmoid

LEDGER = [
    "revenue",
    "sell_fees",
    "pick_pack",
    "purchase",
    "inbound",
    "basket_shipping",
    "disposal",
    "holding",
    "overflow",
    "liquidation",
    "marketing",
]


@dataclass
class SimState:
    """Everything that carries over from one day to the next."""

    # stock
    u_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    u_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    u_grade: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    u_listed: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    u_price: np.ndarray = field(default_factory=lambda: np.zeros(0))
    u_cost: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # on the way to the warehouse
    t_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    t_arrive: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    t_offer: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t_latent: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t_seller: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    t_basket: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    # sellers
    s_factor: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_quality: np.ndarray = field(default_factory=lambda: np.zeros(0))
    s_visits: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    s_first_day: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    r_seller: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # scheduled returns
    r_day: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    # market noise
    feed_e: np.ndarray | None = None
    dump_left: np.ndarray | None = None
    next_unit: int = 1
    next_basket: int = 1
    day: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> SimState:
        return pickle.loads(path.read_bytes())


@dataclass
class Result:
    policy: str
    seed: int
    first_day: int
    days: int
    ledger: dict[str, float]
    daily: pd.DataFrame
    state: SimState
    logs: dict[str, pd.DataFrame] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    price_log: pd.DataFrame | None = None

    @property
    def contribution(self) -> float:
        return float(sum(self.ledger.values()))


class _Log:
    def __init__(self, on: bool):
        self.on = on
        self.rows: dict[str, list[pd.DataFrame]] = {}

    def add(self, table: str, **cols: Any) -> None:
        if self.on:
            self.rows.setdefault(table, []).append(pd.DataFrame(cols))

    def frames(self) -> dict[str, pd.DataFrame]:
        return {k: pd.concat(v, ignore_index=True) for k, v in self.rows.items()}


class Simulator:
    def __init__(self, world: World, seed: int, costs: dict[str, Any] | None = None, log: bool = False, log_prices: bool = False):
        self.w = world
        self.cfg = world.cfg
        self.seed = seed
        self.c = costs or load_costs()
        self.log = _Log(log)
        self.log_prices = log_prices
        self.qf = world.condition_price_factor()
        self.cut = np.array(self.cfg["condition"]["cutpoints"])
        self._ranked = False
        self._sold_yesterday: tuple[np.ndarray, np.ndarray] = (np.zeros(0, int), np.zeros(0))

    # ---------------------------------------------------------------------------------------------
    def run(self, policy: Policy, first_day: int, days: int, state: SimState | None = None, sessions_factor: float = 1.0) -> Result:
        st = state if state is not None else SimState(day=first_day)
        if st.feed_e is None:
            st.feed_e = np.zeros(self.w.n)
            st.dump_left = np.zeros(self.w.n, np.int64)
        ledger = dict.fromkeys(LEDGER, 0.0)
        counters: dict[str, float] = {
            "offers_shown": 0,
            "items_scanned": 0,
            "items_kept": 0,
            "baskets": 0,
            "baskets_sent": 0,
            "new_baskets": 0,
            "new_baskets_sent": 0,
            "units_received": 0,
            "rejects": 0,
            "units_sold": 0,
            "units_liquidated": 0,
            "sell_days_sum": 0,
            "overflow_unit_days": 0,
            "max_fill": 0.0,
            "returns_scheduled": 0,
        }
        daily = []
        price_rows: list[pd.DataFrame] = []
        for day in range(first_day, first_day + days):
            st.day = day
            k = self._day(policy, day, st, ledger, counters, price_rows, sessions_factor)
            daily.append(k)
        res = Result(policy.name, self.seed, first_day, days, ledger, pd.DataFrame(daily), st, self.log.frames(), counters)
        if price_rows:
            res.price_log = pd.concat(price_rows, ignore_index=True)
        return res

    # ---------------------------------------------------------------------------------------------
    def _market(self, day: int, st: SimState) -> MarketUpdate:
        w, f = self.w, self.cfg["market_feed"]
        rng = np.random.default_rng([self.seed, 1, day])
        assert st.feed_e is not None and st.dump_left is not None
        st.feed_e = f["ar"] * st.feed_e + rng.normal(0, f["noise_sd"], w.n)
        start_dump = (rng.random(w.n) < f["dump_rate"]) & (st.dump_left == 0)
        st.dump_left = np.where(start_dump, rng.integers(1, 6, w.n), np.maximum(st.dump_left - 1, 0))
        m = w.m(day)
        released = day >= w.release_day
        seen = w.covered & released & (rng.random(w.n) >= f["missing_day"])
        idx = np.flatnonzero(seen)
        logp = np.log(m[idx]) + st.feed_e[idx] - f["dump_size"] * (st.dump_left[idx] > 0)
        price = np.round(np.exp(logp), 2).clip(0.5)
        lam = w.lam(day)
        offers = rng.poisson(f["offers_base"] + f["offers_scale"] * np.sqrt(lam[idx]))
        u = MarketUpdate(day, idx, price, offers)
        self.log.add("market_feed", day=day, idx=idx, price=price, n_offers=offers)
        d = w.date(day)
        if d.weekday() == 0:
            qi = np.flatnonzero(w.quoted & released)
            q = np.round(w.quote(day)[qi] * np.exp(rng.normal(0, f["quote_noise"], len(qi))), 2).clip(0.01)
            u.quote_idx, u.quote = qi, q
            self.log.add("buyback_quotes", day=day, idx=qi, quote=q)
        if d.day == 1 or not self._ranked:
            self._ranked = True
            ri = np.flatnonzero(released)
            noisy = lam[ri] * np.exp(rng.normal(0, f["rank_noise"], len(ri)))
            order = np.argsort(-noisy)
            rank = np.empty(len(ri))
            rank[order] = (np.arange(len(ri)) + 1) * f["rank_scale"] + rng.integers(0, f["rank_scale"], len(ri))
            u.rank_idx, u.rank = ri, rank
            self.log.add("sales_ranks", day=day, idx=ri, rank=rank)
        return u

    def _warehouse(self, st: SimState) -> Warehouse:
        depth = np.bincount(st.u_idx, minlength=self.w.n) + np.bincount(st.t_idx, minlength=self.w.n)
        return Warehouse(len(st.u_id), len(st.t_idx), self.w.capacity, depth)

    def _day(self, policy: Policy, day: int, st: SimState, L: dict, C: dict, price_rows: list, sessions_factor: float) -> dict:
        w, c = self.w, self.c
        before = sum(L.values())
        upd = self._market(day, st)
        upd.sold_idx, upd.sold_price = self._sold_yesterday
        policy.observe(upd)

        # 1. arrivals: grade, pay, shelve -------------------------------------------------------
        arr = st.t_arrive == day
        if arr.any():
            ai = st.t_idx[arr]
            years = np.maximum(0, w.date(day).year - w.pub_year[ai])
            latent = st.t_latent[arr] + w.grade_shift[ai] + self.cfg["condition"]["age_shift_per_year"] * years + st.s_quality[st.t_seller[arr]]
            grade = np.searchsorted(self.cut, latent)
            rej = grade == REJECT
            L["inbound"] -= c["buy"]["inbound_handling"] * arr.sum()
            L["disposal"] -= c["buy"]["reject_disposal"] * rej.sum()
            ok = ~rej
            L["purchase"] -= st.t_offer[arr][ok].sum()
            n_new = int(ok.sum())
            ids = np.arange(st.next_unit, st.next_unit + n_new)
            st.next_unit += n_new
            C["units_received"] += n_new
            C["rejects"] += int(rej.sum())
            self.log.add(
                "receipts",
                day=day,
                unit_id=_ids_with_rejects(ok, ids),
                basket_id=st.t_basket[arr],
                idx=ai,
                grade=grade,
                paid=np.where(ok, st.t_offer[arr], 0.0),
            )
            st.u_id = np.concatenate([st.u_id, ids])
            st.u_idx = np.concatenate([st.u_idx, ai[ok]])
            st.u_grade = np.concatenate([st.u_grade, grade[ok]])
            st.u_listed = np.concatenate([st.u_listed, np.full(n_new, day)])
            st.u_price = np.concatenate([st.u_price, np.full(n_new, np.nan)])
            st.u_cost = np.concatenate([st.u_cost, st.t_offer[arr][ok]])
            keep = ~arr
            for a in ("t_idx", "t_arrive", "t_offer", "t_latent", "t_seller", "t_basket"):
                setattr(st, a, getattr(st, a)[keep])

        # 2. pricing and liquidation ---------------------------------------------------------------
        wh = self._warehouse(st)
        reprice = w.date(day).weekday() == 0
        needs = np.isnan(st.u_price) | reprice
        if len(st.u_id):
            units = Units(st.u_id.copy(), st.u_idx.copy(), st.u_grade.copy(), st.u_listed.copy(), st.u_price.copy(), needs)
            dec = policy.price(day, units, wh)
            newp = np.where(needs, dec.price, st.u_price)
            if not np.all(np.isfinite(newp) & (newp > 0)):
                raise ValueError(f"{policy.name}: invalid price on day {day}")
            changed = needs & (np.isnan(st.u_price) | (np.abs(newp - np.nan_to_num(st.u_price)) > 1e-9))
            if self.log.on and changed.any():
                self.log.add(
                    "prices",
                    day=day,
                    unit_id=st.u_id[changed],
                    price=np.round(newp[changed], 2),
                    explore=(dec.explore[changed] if dec.explore is not None else np.full(changed.sum(), np.nan)),
                )
            if self.log_prices and needs.any():
                price_rows.append(
                    pd.DataFrame(
                        {
                            "day": day,
                            "unit_id": st.u_id[needs],
                            "idx": st.u_idx[needs],
                            "grade": st.u_grade[needs],
                            "old": st.u_price[needs],
                            "price": newp[needs],
                            "age": day - st.u_listed[needs],
                        }
                    )
                )
            st.u_price = np.round(newp, 2)
            liq = dec.liquidate | (day - st.u_listed >= c["stock"]["max_age_days"])
            if liq.any():
                L["liquidation"] += (c["stock"]["liquidation_value"] - c["stock"]["liquidation_handling"]) * liq.sum()
                C["units_liquidated"] += int(liq.sum())
                self.log.add("disposals", day=day, unit_id=st.u_id[liq], kind="liquidation")
                self._drop(st, ~liq)

        # 3. buyers --------------------------------------------------------------------------------
        sold_idx, sold_price = np.zeros(0, int), np.zeros(0)
        if len(st.u_id):
            rng = np.random.default_rng([self.seed, 5, day])
            m, lam = w.m(day), w.lam(day)
            n_i = np.bincount(st.u_idx, minlength=w.n)
            rel = np.log(st.u_price / (m[st.u_idx] * self.qf[st.u_grade]))
            intensity = lam[st.u_idx] * w.share(rel, st.u_idx) / n_i[st.u_idx]
            sold = rng.random(len(st.u_id)) < 1 - np.exp(-intensity)
            if sold.any():
                p = st.u_price[sold]
                fees = c["sell"]["fee_rate"] * p + c["sell"]["fee_fixed"]
                L["revenue"] += p.sum()
                L["sell_fees"] -= fees.sum()
                L["pick_pack"] -= c["sell"]["pick_pack"] * sold.sum()
                C["units_sold"] += int(sold.sum())
                C["sell_days_sum"] += int((day - st.u_listed[sold]).sum())
                self.log.add("sales", day=day, unit_id=st.u_id[sold], price=p, fee=np.round(fees, 2))
                sold_idx, sold_price = st.u_idx[sold], p
                self._drop(st, ~sold)
        self._sold_yesterday = (sold_idx, sold_price)

        # 4. warehouse costs -----------------------------------------------------------------------
        L["holding"] -= c["stock"]["holding_per_unit_day"] * len(st.u_id)
        over = max(0, len(st.u_id) - w.capacity)
        L["overflow"] -= self.cfg["warehouse"]["overflow_cost_per_unit_day"] * over
        C["overflow_unit_days"] += over
        fill = len(st.u_id) / w.capacity
        C["max_fill"] = max(C["max_fill"], fill)

        # 5. sellers -------------------------------------------------------------------------------
        bought = self._sellers(policy, day, st, L, C, sessions_factor)

        return {
            "day": day,
            "stock": len(st.u_id),
            "in_transit": len(st.t_idx),
            "fill": round(fill, 4),
            "sold": len(sold_idx),
            "sales_eur": round(float(sold_price.sum()), 2),
            "bought": bought,
            "cash": round(sum(L.values()) - before, 2),
        }

    def _drop(self, st: SimState, keep: np.ndarray) -> None:
        for a in ("u_id", "u_idx", "u_grade", "u_listed", "u_price", "u_cost"):
            setattr(st, a, getattr(st, a)[keep])

    # ---------------------------------------------------------------------------------------------
    def _sellers(self, policy: Policy, day: int, st: SimState, L: dict, C: dict, sessions_factor: float) -> int:
        w, s, c = self.w, self.cfg["sellers"], self.c
        rng = np.random.default_rng([self.seed, 2, day])
        supply_season = float(bumps(self.cfg["season"]["supply"], w.doy(day)))
        supply_override = float(np.mean(w._override_factor(day, "sessions"))) if w.overrides else 1.0
        n_new = rng.poisson(s["new_sessions_per_day"] * supply_season * supply_override * sessions_factor)
        L["marketing"] -= c["marketing"]["cost_per_new_session"] * n_new
        # new sellers
        new_ids = np.arange(len(st.s_factor), len(st.s_factor) + n_new)
        st.s_factor = np.concatenate([st.s_factor, np.exp(rng.normal(0, s["seller_factor_sigma"], n_new))])
        st.s_quality = np.concatenate([st.s_quality, rng.normal(0, self.cfg["condition"]["seller_quality_sigma"], n_new)])
        st.s_visits = np.concatenate([st.s_visits, np.zeros(n_new, np.int64)])
        st.s_first_day = np.concatenate([st.s_first_day, np.full(n_new, day)])
        due = st.r_day == day
        ret_ids = st.r_seller[due]
        st.r_seller, st.r_day = st.r_seller[~due], st.r_day[~due]
        sellers = np.concatenate([new_ids, ret_ids]).astype(np.int64)
        if len(sellers) == 0:
            return 0
        supply = w.supply_weights(day)
        cum = np.cumsum(supply)
        # each seller visit gets its own random stream: same items whatever the policy
        b_rows = []
        for bnum, sid in enumerate(sellers):
            visit = int(st.s_visits[sid])
            r = np.random.default_rng([self.seed, 3, int(sid), visit])
            size = 1 + r.negative_binomial(s["basket_dispersion"], s["basket_dispersion"] / (s["basket_dispersion"] + s["basket_mean_items"] - 1))
            items = np.searchsorted(cum, r.random(size) * cum[-1]).clip(0, w.n - 1)
            items = np.unique(items)
            res_noise = np.exp(r.normal(0, s["reservation_sigma"], len(items)))
            latent = r.standard_normal(len(items))
            ship = r.integers(s["shipping_days"][0], s["shipping_days"][1] + 1)
            u_ret = r.random()
            delay = r.integers(s["return_delay_days"][0], s["return_delay_days"][1] + 1)
            b_rows.append((bnum, sid, visit, items, res_noise, latent, ship, u_ret, delay))
        basket = np.concatenate([np.full(len(b[3]), b[0]) for b in b_rows])
        seller = np.concatenate([np.full(len(b[3]), b[1]) for b in b_rows])
        visit = np.concatenate([np.full(len(b[3]), b[2]) for b in b_rows])
        idx = np.concatenate([b[3] for b in b_rows]).astype(np.int64)
        items = Items(basket, seller, visit, idx)
        dec = policy.offer(day, items, self._warehouse(st))
        offer = np.round(np.where(dec.offer >= c["buy"]["min_offer"], dec.offer, 0.0), 2)
        q = w.quote(day)[idx]
        reservation = q * st.s_factor[seller] * np.concatenate([b[4] for b in b_rows])
        kept = (offer > 0) & (offer >= reservation)
        n_b = len(b_rows)
        offered_sum = np.bincount(basket, weights=offer, minlength=n_b)
        quote_sum = np.bincount(basket, weights=np.where(offer > 0, q, 0.0), minlength=n_b)
        kept_sum = np.bincount(basket, weights=np.where(kept, offer, 0.0), minlength=n_b)
        sent = kept_sum >= s["min_basket_eur"]
        sat = np.where(quote_sum > 0, offered_sum / np.maximum(quote_sum, 1e-9), s["no_offer_satisfaction"])
        C["items_scanned"] += len(idx)
        C["offers_shown"] += int((offer > 0).sum())
        C["baskets"] += n_b
        C["baskets_sent"] += int(sent.sum())
        is_new = np.array([b[2] == 0 for b in b_rows])
        C["new_baskets"] += int(is_new.sum())
        C["new_baskets_sent"] += int((is_new & sent).sum())

        basket_ids = np.arange(st.next_basket, st.next_basket + n_b)
        st.next_basket += n_b
        if self.log.on:
            self.log.add(
                "offers",
                day=day,
                basket_id=basket_ids[basket],
                seller_id=seller,
                idx=idx,
                offer=offer,
                explore=dec.explore if dec.explore is not None else np.full(len(idx), np.nan),
                kept=kept,
            )
            self.log.add(
                "baskets",
                day=day,
                basket_id=basket_ids,
                seller_id=sellers,
                visit=np.array([b[2] for b in b_rows]),
                items=np.bincount(basket, minlength=n_b),
                offered_eur=np.round(offered_sum, 2),
                kept_eur=np.round(kept_sum, 2),
                sent=sent,
            )
        # parcels on their way
        go = kept & sent[basket]
        if go.any():
            ship = np.array([b[6] for b in b_rows])[basket[go]]
            st.t_idx = np.concatenate([st.t_idx, idx[go]])
            st.t_arrive = np.concatenate([st.t_arrive, day + ship])
            st.t_offer = np.concatenate([st.t_offer, offer[go]])
            st.t_latent = np.concatenate([st.t_latent, np.concatenate([b[5] for b in b_rows])[go]])
            st.t_seller = np.concatenate([st.t_seller, seller[go]])
            st.t_basket = np.concatenate([st.t_basket, basket_ids[basket[go]]])
        L["basket_shipping"] -= c["buy"]["basket_shipping"] * sent.sum()
        C["items_kept"] += int(go.sum())

        # will they come back?
        logit = np.where(
            sent,
            s["return_intercept"] + s["return_satisfaction"] * np.log(np.maximum(sat, 1e-3)) + s["return_value"] * np.log1p(kept_sum / 10),
            s["return_intercept_abandoned"] + s["return_satisfaction"] * np.log(np.maximum(sat, 1e-3)),
        )
        p_ret = sigmoid(logit)
        u_ret = np.array([b[7] for b in b_rows])
        back = u_ret < p_ret
        delays = np.array([b[8] for b in b_rows])
        st.r_seller = np.concatenate([st.r_seller, sellers[back]])
        st.r_day = np.concatenate([st.r_day, day + delays[back]])
        C["returns_scheduled"] += int(back.sum())
        st.s_visits[sellers] += 1
        return int(go.sum())


def _ids_with_rejects(ok: np.ndarray, ids: np.ndarray) -> np.ndarray:
    out = np.full(len(ok), -1, dtype=np.int64)
    out[ok] = ids
    return out
