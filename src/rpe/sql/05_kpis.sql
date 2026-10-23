-- Weekly KPI tree for steering: buying, selling, stock, margin.
create or replace table kpi_weekly as
with wk as (select day, (day - (day % 7)) as week from range(0, 2000) r(day)),
buy as (
    select (o.day - o.day % 7) as week, count(*) as items_scanned, count(*) filter (where o.offer > 0) as offers,
           count(*) filter (where o.kept) as kept, sum(o.offer) filter (where o.kept) as kept_eur
    from offers o group by 1),
bsk as (
    select (day - day % 7) as week, count(*) as baskets, count(*) filter (where sent) as baskets_sent
    from baskets group by 1),
rec as (
    select (day - day % 7) as week, count(*) as received, count(*) filter (where grade = 'reject') as rejects,
           sum(paid_eur) as purchase_eur
    from receipts group by 1),
sal as (
    select (sold_day - sold_day % 7) as week, count(*) as units_sold, sum(sold_price) as revenue, sum(fee_eur) as fees,
           avg(sold_day - listed_day) as days_to_sell, sum(paid_eur) as cogs
    from units where sold_day is not null group by 1),
liq as (
    select (disposed_day - disposed_day % 7) as week, count(*) as liquidated from units where disposed_day is not null group by 1),
stk as (
    select (day - day % 7) as week, avg(stock_units) as stock, max(stock_units::double / capacity_units) as max_fill,
           avg(capacity_units) as capacity
    from warehouse_daily group by 1)
select stk.week, stk.stock, stk.max_fill, stk.capacity,
       buy.items_scanned, buy.offers, buy.kept, buy.kept_eur, bsk.baskets, bsk.baskets_sent,
       rec.received, rec.rejects, rec.purchase_eur,
       sal.units_sold, sal.revenue, sal.fees, sal.days_to_sell, sal.cogs, coalesce(liq.liquidated, 0) as liquidated
from stk
left join buy using (week) left join bsk using (week) left join rec using (week)
left join sal using (week) left join liq using (week)
order by week;
