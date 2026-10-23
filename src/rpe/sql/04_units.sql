-- One row per unit we ever held: where it came from, what it cost, how it left.
create or replace table units as
with u as (
    select unit_id, ean, grade, date_diff('day', DATE '{epoch}', listed_date::date) as listed_day, paid_eur, true as opening
    from read_parquet('{clean}/opening_stock.parquet')
    union all
    select unit_id, ean, grade, date_diff('day', DATE '{epoch}', date::date), paid_eur, false
    from read_parquet('{clean}/receipts.parquet') where unit_id is not null
)
select u.unit_id, t.idx, u.grade,
       case u.grade when 'like_new' then 0 when 'very_good' then 1 when 'good' then 2 else 3 end as grade_code,
       u.listed_day, u.paid_eur, u.opening,
       date_diff('day', DATE '{epoch}', s.date::date) as sold_day, s.price_eur as sold_price, s.fee_eur,
       date_diff('day', DATE '{epoch}', d.date::date) as disposed_day
from u join dim_title t using (ean)
left join read_parquet('{clean}/sales.parquet') s using (unit_id)
left join read_parquet('{clean}/disposals.parquet') d using (unit_id);

create or replace table prices as
select date_diff('day', DATE '{epoch}', p.date::date) as day, p.unit_id, p.price_eur as price, p.test_multiplier
from read_parquet('{clean}/prices.parquet') p;

create or replace table warehouse_daily as
select date_diff('day', DATE '{epoch}', date::date) as day, stock_units, in_transit_units, capacity_units
from read_parquet('{clean}/warehouse_daily.parquet');
