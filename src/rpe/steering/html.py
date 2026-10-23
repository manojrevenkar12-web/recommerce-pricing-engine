"""Self-contained HTML for the steering report (no scripts, no external assets; charts are SVG)."""

from __future__ import annotations

import html
from typing import Any

CSS = """
:root{--paper:#f4f6f3;--ink:#16302f;--muted:#5b6f6c;--rule:#d5ddd9;--good:#2a7a5f;--bad:#b23a48;--warn:#a86b12;--bar:#3f6f6a;--card:#ffffff}
@media (prefers-color-scheme: dark){:root{--paper:#101716;--ink:#e3ece9;--muted:#9bb0ab;--rule:#2a3836;--good:#5cc29c;--bad:#e27a86;--warn:#e0a64a;--bar:#7fb3ab;--card:#16201f}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,"Segoe UI",Helvetica,Arial,sans-serif;font-variant-numeric:tabular-nums}
main{max-width:1040px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:26px;line-height:1.2;margin:0 0 4px;font-weight:650;letter-spacing:-.01em}
h2{font-size:17px;margin:36px 0 12px;font-weight:650}
.sub{color:var(--muted);margin:0 0 24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule)}
.kpi{background:var(--card);padding:14px 16px}.kpi .v{font-size:24px;font-weight:650}.kpi .l{color:var(--muted);font-size:13px}
.up{color:var(--good)}.down{color:var(--bad)}
.alerts{list-style:none;padding:0;margin:0}.alerts li{background:var(--card);border-left:4px solid var(--warn);padding:10px 14px;margin:0 0 8px}
.alerts li.red{border-left-color:var(--bad)}.alerts b{display:block}.alerts span{color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}
figure{margin:0;background:var(--card);padding:14px 16px;border:1px solid var(--rule)}figcaption{font-size:13px;color:var(--muted);margin-bottom:6px}
svg text{fill:var(--muted);font-size:11px}
table{border-collapse:collapse;width:100%;background:var(--card)}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);font-size:14px}
th{color:var(--muted);font-weight:600}td.n{text-align:right}
.barcell{width:40%}.barcell div{height:10px;background:var(--bar)}
footer{color:var(--muted);font-size:13px;margin-top:40px}
"""


def _line(
    points: list[float], points2: list[float] | None, labels: list[str], w: int = 460, h: int = 160, names: tuple[str, str] = ("", "")
) -> str:
    vals = points + (points2 or [])
    lo, hi = min(0.0, min(vals)), max(vals) * 1.08 or 1.0
    n = len(points)

    def xy(i: int, v: float) -> tuple[float, float]:
        return 36 + i * (w - 48) / max(n - 1, 1), 10 + (h - 34) * (1 - (v - lo) / (hi - lo))

    def path(p: list[float]) -> str:
        return " ".join(("M" if i == 0 else "L") + f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in enumerate(p))

    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(names[0])}">']
    for t in (lo, (lo + hi) / 2, hi):
        y = xy(0, t)[1]
        out.append(
            f'<line x1="36" x2="{w - 12}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--rule)"/><text x="2" y="{y + 4:.1f}">{t:,.0f}</text>'
        )
    out.append(f'<path d="{path(points)}" fill="none" stroke="var(--ink)" stroke-width="2"/>')
    if points2:
        out.append(f'<path d="{path(points2)}" fill="none" stroke="var(--good)" stroke-width="2" stroke-dasharray="4 3"/>')
    for i in (0, n - 1):
        out.append(f'<text x="{xy(i, lo)[0] - (40 if i else 0):.1f}" y="{h - 4}">{html.escape(labels[i])}</text>')
    out.append("</svg>")
    legend = (
        f'<div style="font-size:12px;color:var(--muted)">— {html.escape(names[0])}'
        + (f" &nbsp; - - {html.escape(names[1])}" if points2 else "")
        + "</div>"
    )
    return "".join(out) + legend


def _bars(rows: list[tuple[str, float]], w: int = 460) -> str:
    h = 22 * len(rows) + 10
    m = max(abs(v) for _, v in rows) or 1.0
    mid = 170 + (w - 190) / 2
    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="competitor price change by category">',
        f'<line x1="{mid}" x2="{mid}" y1="0" y2="{h}" stroke="var(--rule)"/>',
    ]
    for i, (name, v) in enumerate(rows):
        y = 6 + i * 22
        bw = abs(v) / m * (w - 190) / 2
        x = mid if v >= 0 else mid - bw
        color = "var(--bad)" if v < -5 else "var(--bar)"
        out.append(
            f'<text x="0" y="{y + 11}">{html.escape(name)}</text><rect x="{x:.1f}" y="{y}" width="{bw:.1f}" height="14" fill="{color}"/>'
        )
        out.append(f'<text x="{(mid + bw + 4) if v >= 0 else (mid - bw - 44):.1f}" y="{y + 11}">{v:+.1f}%</text>')
    out.append("</svg>")
    return "".join(out)


def _kpi(label: str, value: str, change: float | None) -> str:
    ch = ""
    if change is not None:
        cls = "up" if change >= 0 else "down"
        ch = f' <span class="{cls}">{change:+.1f}%</span>'
    return f'<div class="kpi"><div class="l">{html.escape(label)}</div><div class="v">{value}{ch}</div></div>'


def render_html(r: dict[str, Any]) -> str:
    k = r["kpis"]
    wk = r["weekly"]
    labels = [w["week"] for w in wk]
    alerts = (
        "".join(
            f'<li class="{a["level"]}"><b>{html.escape(a["what"])}</b>{html.escape(a["detail"])}. <span>{html.escape(a["action"])}</span></li>'
            for a in r["alerts"]
        )
        or "<li>No alerts this week.</li>"
    )
    road = ""
    if r["roadmap"]:
        m = max(abs(u["value_eur"]) for u in r["roadmap"]) or 1
        road = (
            "<h2>AI roadmap, ranked by measured value</h2><table><tr><th>Use case</th><th>Value, six months</th><th class='barcell'></th></tr>"
            + "".join(
                f"<tr><td>{html.escape(u['use_case'])}</td><td class='n'>EUR {u['value_eur']:,.0f}<br><span style='color:var(--muted);font-size:12px'>{u['lo']:,.0f} to {u['hi']:,.0f}</span></td>"
                f"<td class='barcell'><div style='width:{max(0, u['value_eur']) / m * 100:.0f}%'></div></td></tr>"
                for u in r["roadmap"]
            )
            + "</table>"
        )
    bt = ""
    if r.get("engine_backtest"):
        h = r["engine_backtest"]
        bt = (
            f"<h2>Pricing engine vs today's rules</h2><p>Six-month backtest on {h['seeds']} market replays: value (cash plus the value of stock at the end) "
            f"<b class='up'>{h['value_pct']:+.1f}%</b> (95% interval {h['value_pct_lo']:+.1f}% to {h['value_pct_hi']:+.1f}%); cash within the period {h['cash_pct']:+.1f}%; "
            f"sellers who will come back {h['returns_pct']:+.1f}%.</p>"
        )
    exps = "".join(
        f"<tr><td><code>{html.escape(e['id'])}</code></td><td>{html.escape(e['unit'])}</td><td>{html.escape(e['hypothesis'])}</td></tr>"
        for e in r["experiments"]
    )
    body = f"""
<main>
<h1>Books &amp; Media pricing</h1>
<p class="sub">Weekly steering, week ending {html.escape(r["as_of"])}. Last 4 weeks against the 4 before.</p>
<section class="kpis">
{_kpi("Revenue", f"EUR {k['revenue_eur']['last_4_weeks']:,.0f}", k["revenue_eur"]["change_pct"])}
{_kpi("Gross margin", f"EUR {k['gross_margin_eur']['last_4_weeks']:,.0f}", k["gross_margin_eur"]["change_pct"])}
{_kpi("Contribution", f"EUR {k['contribution_eur']['last_4_weeks']:,.0f}", k["contribution_eur"]["change_pct"])}
{_kpi("Units sold", f"{k['units_sold']['last_4_weeks']:,.0f}", k["units_sold"]["change_pct"])}
{_kpi("Offers kept by sellers", f"{k['seller_keep_rate']['last_4_weeks']:.1%}", None)}
{_kpi("Average sale price", f"EUR {k['avg_sale_price_eur']['last_4_weeks']:.2f}", None)}
{_kpi("Average paid per unit", f"EUR {k['avg_paid_eur']['last_4_weeks']:.2f}", None)}
{_kpi("Warehouse fill", f"{k['warehouse_fill']:.0%}", None)}
</section>
<h2>Alerts</h2>
<ul class="alerts">{alerts}</ul>
<h2>Trends</h2>
<div class="grid">
<figure><figcaption>Revenue and contribution per week, EUR</figcaption>{_line([w["revenue"] for w in wk], [w["contribution"] for w in wk], labels, names=("revenue", "contribution"))}</figure>
<figure><figcaption>Units in stock per week</figcaption>{_line([w["stock"] for w in wk], None, labels, names=("stock", ""))}</figure>
<figure><figcaption>Competitor lowest prices, change over 4 weeks</figcaption>{_bars([(p["category"], p["change_pct"]) for p in r["category_prices"]])}</figure>
</div>
{bt}
{road}
<h2>Experiments</h2>
<table><tr><th>Experiment</th><th>Randomised by</th><th>Hypothesis</th></tr>{exps}</table>
<footer>Generated by <code>rpe steering</code> from the warehouse. Figures are from simulated data.</footer>
</main>"""
    return f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Pricing steering</title><style>{CSS}</style></head><body>{body}</body></html>"
