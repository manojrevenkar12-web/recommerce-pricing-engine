create or replace table offers as
select o.offer_id, date_diff('day', DATE '{epoch}', o.date::date) as day, o.basket_id, o.seller_id, t.idx,
       o.offer_eur as offer, o.test_multiplier, o.kept
from read_parquet('{clean}/offers.parquet') o join dim_title t using (ean);

create or replace table baskets as
select basket_id, date_diff('day', DATE '{epoch}', date::date) as day, seller_id, visit, items, offered_eur, kept_eur, sent
from read_parquet('{clean}/baskets.parquet');

create or replace table receipts as
select date_diff('day', DATE '{epoch}', r.date::date) as day, r.basket_id, t.idx, r.unit_id, r.grade, r.paid_eur
from read_parquet('{clean}/receipts.parquet') r join dim_title t using (ean);
