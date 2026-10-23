"""Weekly steering report for Books & Media business-unit management.

Reads only the warehouse (what the company knows) and, for the AI roadmap, the measured value of
each engine component from the evaluation. Writes reports/steering.{json,md,html}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rpe.calendar import to_date
from rpe.catalog import CATEGORIES
from rpe.config import REPORTS, WAREHOUSE, costs, load
from rpe.steering.html import render_html
from rpe.warehouse import query


def weekly(wh: Path = WAREHOUSE) -> pd.DataFrame:
    c = costs()
    k = query("select * from kpi_weekly where items_scanned is not null order by week", wh)
    k = k[k.week >= k.week.min() + 7].copy()  # first partial week
    k["offer_rate"] = k.offers / k.items_scanned
    k["keep_rate"] = k.kept / k.offers
    k["sent_rate"] = k.baskets_sent / k.baskets
    k["avg_paid"] = k.purchase_eur / (k.received - k.rejects)
    k["avg_sale_price"] = k.revenue / k.units_sold
    k["gross_margin"] = k.revenue - k.cogs
    k["contribution"] = (
        k.revenue
        - k.fees
        - c["sell"]["pick_pack"] * k.units_sold
        - k.purchase_eur
        - c["buy"]["inbound_handling"] * k.received
        - c["buy"]["basket_shipping"] * k.baskets_sent
        - 7 * c["stock"]["holding_per_unit_day"] * k.stock
    )
    k["fill"] = k["max_fill"]
    k["sell_through"] = k.units_sold / k.stock
    k["margin_per_unit"] = k.gross_margin / k.units_sold
    return k.reset_index(drop=True)


def category_price_index(wh: Path = WAREHOUSE) -> pd.DataFrame:
    """Median 28-day change of competitor lowest prices per category (titles seen in both windows)."""
    last = int(query("select max(day) as d from feed", wh).d[0])
    q = query(
        f"""
        with a as (select idx, median(price) as p from feed where day > {last - 7} group by 1),
             b as (select idx, median(price) as p from feed where day between {last - 35} and {last - 28} group by 1)
        select t.category, median(ln(a.p / b.p)) as change, count(*) as titles
        from a join b using (idx) join dim_title t using (idx) group by 1 order by 1
        """,
        wh,
    )
    q["change_pct"] = (np.exp(q.change) - 1) * 100
    return q


def aged_stock(wh: Path = WAREHOUSE) -> dict[str, float]:
    last = int(query("select max(day) as d from warehouse_daily", wh).d[0])
    q = query(
        f"""select count(*) as n, count(*) filter (where {last} - listed_day > 180) as aged,
                   count(*) filter (where {last} - listed_day > 365) as over_a_year
            from units where coalesce(sold_day, 99999) > {last} and coalesce(disposed_day, 99999) > {last}""",
        wh,
    )
    n = int(q.n[0])
    return {"units": n, "aged_180d_share": float(q.aged[0]) / max(n, 1), "over_365d_share": float(q.over_a_year[0]) / max(n, 1)}


def alerts(k: pd.DataFrame, prices: pd.DataFrame, aged: dict[str, float], cfg: dict[str, Any]) -> list[dict[str, str]]:
    a = cfg["alerts"]
    last4, prev4 = k.tail(4), k.iloc[-8:-4]
    out = []

    def add(level: str, what: str, detail: str, action: str) -> None:
        out.append({"level": level, "what": what, "detail": detail, "action": action})

    dk = (last4.kept.sum() / last4.offers.sum() - prev4.kept.sum() / prev4.offers.sum()) * 100
    if dk < -a["acceptance_drop_pp"]:
        add(
            "red",
            "Sellers keep fewer offers",
            f"{dk:+.1f} pp vs the previous 4 weeks",
            "Check competitor buy-back quotes; review offer levels for the categories that fell most.",
        )
    fill = float(k.fill.iloc[-1])
    if fill > a["fill_above"]:
        add(
            "amber",
            "Warehouse nearly full",
            f"fill {fill:.0%}",
            "Expect the shelf price to rise: fewer slow sellers bought, faster markdowns and liquidation.",
        )
    if aged["aged_180d_share"] > a["aged_stock_share_above"]:
        add(
            "amber",
            "Old stock",
            f"{aged['aged_180d_share']:.0%} of stock listed for more than 180 days",
            "Review liquidation of copies whose expected value is below bulk value.",
        )
    m_last = last4.gross_margin.sum() / last4.units_sold.sum()
    m_prev = prev4.gross_margin.sum() / prev4.units_sold.sum()
    if m_last < m_prev * (1 - a["margin_per_unit_drop"]):
        add(
            "red",
            "Margin per unit falling",
            f"EUR {m_last:.2f} vs {m_prev:.2f}",
            "Look for categories where prices fell faster than buy offers.",
        )
    st_last = last4.units_sold.sum() / last4.stock.sum()
    st_prev = prev4.units_sold.sum() / prev4.stock.sum()
    if st_last < st_prev * (1 - a["sell_through_drop"]):
        add(
            "amber",
            "Selling slower",
            f"sell-through {st_last:.3f} vs {st_prev:.3f} per week",
            "Check whether prices are above the market in the categories that slowed.",
        )
    for r in prices.itertuples():
        if r.change_pct < -100 * a["category_price_drop"]:
            add(
                "red",
                f"Price war watch: {r.category}",
                f"competitor prices {r.change_pct:+.1f}% in 4 weeks ({r.titles} titles)",
                "Lower buy offers for the category now; the market price model follows within a week.",
            )
    return out


def roadmap(evaluation: dict[str, Any] | None) -> list[dict[str, Any]]:
    """AI use cases ranked by the value measured in the backtest (engine with vs without the part)."""
    if not evaluation:
        return []
    ab = evaluation.get("backtest", {}).get("ablations", {})
    names = {
        "engine_sell_only": ("Buy offers and buy/no-buy (sellability, acceptance, copies in stock)", "buy"),
        "engine_buy_only": ("Dynamic sell prices and liquidation (demand and price response)", "sell"),
        "engine_no_market_model": ("Market price definition (robust feed price, cold start for new titles)", "market"),
        "engine_no_capacity": ("Warehouse-aware buying and selling (shelf price)", "capacity"),
        "engine_no_long_term": ("Long-term seller value in offers", "long_term"),
    }
    out = []
    for k, (label, key) in names.items():
        if k in ab:
            v = ab[k]
            out.append(
                {
                    "use_case": label,
                    "key": key,
                    "value_eur": v["value_of_part"],
                    "lo": v["lo"],
                    "hi": v["hi"],
                    "incl_sellers_eur": v.get("value_of_part_incl_sellers"),
                }
            )
    return sorted(out, key=lambda r: -r["value_eur"])


def build(wh: Path = WAREHOUSE, out: Path = REPORTS) -> dict[str, Any]:
    cfg = load("steering")
    k = weekly(wh)
    prices = category_price_index(wh)
    aged = aged_stock(wh)
    ev_path = out / "evaluation.json"
    evaluation = json.loads(ev_path.read_text()) if ev_path.exists() else None
    shown = k.tail(cfg["weeks_shown"])
    last4, prev4 = k.tail(4), k.iloc[-8:-4]

    def kpi(col: str, agg: str = "sum") -> dict[str, float | None]:
        f = (lambda d: d[col].sum()) if agg == "sum" else (lambda d: d[col].mean())
        a, b = float(f(last4)), float(f(prev4))
        return {"last_4_weeks": round(a, 4), "previous_4_weeks": round(b, 4), "change_pct": round((a / b - 1) * 100, 1) if b else None}

    ratio = lambda num, den: {  # noqa: E731
        "last_4_weeks": round(float(last4[num].sum() / last4[den].sum()), 4),
        "previous_4_weeks": round(float(prev4[num].sum() / prev4[den].sum()), 4),
    }
    report = {
        "as_of": to_date(int(k.week.iloc[-1]) + 6).isoformat(),
        "kpis": {
            "revenue_eur": kpi("revenue"),
            "gross_margin_eur": kpi("gross_margin"),
            "contribution_eur": kpi("contribution"),
            "units_sold": kpi("units_sold"),
            "units_bought": kpi("received"),
            "avg_sale_price_eur": ratio("revenue", "units_sold"),
            "avg_paid_eur": ratio("purchase_eur", "received"),
            "offer_rate": ratio("offers", "items_scanned"),
            "seller_keep_rate": ratio("kept", "offers"),
            "basket_sent_rate": ratio("baskets_sent", "baskets"),
            "stock_units": kpi("stock", "mean"),
            "warehouse_fill": round(float(k.fill.iloc[-1]), 4),
            "liquidated_units": kpi("liquidated"),
        },
        "aged_stock": {kk: round(v, 4) for kk, v in aged.items()},
        "category_prices": prices[["category", "change_pct", "titles"]].round(2).to_dict("records"),
        "alerts": alerts(k, prices, aged, cfg),
        "weekly": shown[
            [
                "week",
                "revenue",
                "gross_margin",
                "contribution",
                "units_sold",
                "received",
                "stock",
                "fill",
                "keep_rate",
                "avg_sale_price",
                "avg_paid",
            ]
        ]
        .round(4)
        .assign(week=lambda d: [to_date(int(x)).isoformat() for x in d.week])
        .to_dict("records"),
        "roadmap": roadmap(evaluation),
        "engine_backtest": (evaluation or {}).get("backtest", {}).get("headline"),
        "experiments": load("experiments"),
        "categories": CATEGORIES,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "steering.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "steering.md").write_text(render_markdown(report))
    (out / "steering.html").write_text(render_html(report))
    return report


def _eur(x: float) -> str:
    return f"EUR {x:,.0f}"


def render_markdown(r: dict[str, Any]) -> str:
    k = r["kpis"]
    L = [
        f"# Books & Media pricing: weekly steering, week ending {r['as_of']}",
        "",
        "## Where we stand (last 4 weeks vs the 4 before)",
        "",
        "| | last 4 weeks | previous 4 | change |",
        "|---|---|---|---|",
    ]
    for key, label in [
        ("revenue_eur", "Revenue"),
        ("gross_margin_eur", "Gross margin (revenue - purchase cost of units sold)"),
        ("contribution_eur", "Contribution (after fees, handling, parcels, holding)"),
        ("units_sold", "Units sold"),
        ("units_bought", "Units received"),
    ]:
        v = k[key]
        L.append(f"| {label} | {v['last_4_weeks']:,.0f} | {v['previous_4_weeks']:,.0f} | {v['change_pct']:+.1f}% |")
    for key, label, fmt in [
        ("avg_sale_price_eur", "Average sale price", "{:.2f}"),
        ("avg_paid_eur", "Average paid per unit", "{:.2f}"),
        ("seller_keep_rate", "Offers kept by sellers", "{:.1%}"),
        ("basket_sent_rate", "Baskets sent", "{:.1%}"),
        ("offer_rate", "Scanned items we offer for", "{:.1%}"),
    ]:
        v = k[key]
        L.append(f"| {label} | {fmt.format(v['last_4_weeks'])} | {fmt.format(v['previous_4_weeks'])} | |")
    L += [
        "",
        f"Stock: {k['stock_units']['last_4_weeks']:,.0f} units, warehouse fill {k['warehouse_fill']:.0%}; "
        f"{r['aged_stock']['aged_180d_share']:.0%} of stock listed for more than 180 days.",
        "",
        "## Alerts",
        "",
    ]
    L += [f"- **{a['what']}** ({a['detail']}). {a['action']}" for a in r["alerts"]] or ["- None this week."]
    L += ["", "## Market watch: competitor prices over 4 weeks", "", "| category | change | titles |", "|---|---|---|"]
    L += [f"| {p['category']} | {p['change_pct']:+.1f}% | {p['titles']} |" for p in r["category_prices"]]
    if r["engine_backtest"]:
        h = r["engine_backtest"]
        L += [
            "",
            "## The pricing engine against today's rules",
            "",
            f"Backtest over six months on {h['seeds']} market replays: value (cash plus the value of stock at the end) "
            f"{h['value_pct']:+.1f}% (95% interval {h['value_pct_lo']:+.1f}% to {h['value_pct_hi']:+.1f}%), cash in the period "
            f"{h['cash_pct']:+.1f}%; sellers who will come back {h['returns_pct']:+.1f}%.",
        ]
    if r["roadmap"]:
        L += ["", "## AI roadmap, ranked by measured value (six months, EUR)", "", "| use case | value | 95% interval |", "|---|---|---|"]
        L += [f"| {u['use_case']} | {u['value_eur']:,.0f} | {u['lo']:,.0f} to {u['hi']:,.0f} |" for u in r["roadmap"]]
    L += ["", "## Experiments", ""]
    L += [f"- `{e['id']}` (by {e['unit']}): {e['hypothesis']}" for e in r["experiments"]]
    return "\n".join(L) + "\n"
