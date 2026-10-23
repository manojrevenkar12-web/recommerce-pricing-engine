create or replace table feed as
select date_diff('day', DATE '{epoch}', f.date::date) as day, t.idx, f.lowest_used_price as price, f.n_offers
from read_parquet('{clean}/market_feed.parquet') f join dim_title t using (ean);

create or replace table quotes as
select date_diff('day', DATE '{epoch}', q.date::date) as day, t.idx, q.competitor_quote as quote
from read_parquet('{clean}/buyback_quotes.parquet') q join dim_title t using (ean);

create or replace table ranks as
select date_diff('day', DATE '{epoch}', r.date::date) as day, t.idx, r.sales_rank as rank
from read_parquet('{clean}/sales_ranks.parquet') r join dim_title t using (ean);
