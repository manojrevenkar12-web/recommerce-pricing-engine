-- Titles get a dense index (in EAN order) that every model and the engine use.
create or replace table dim_title as
select
    (row_number() over (order by ean) - 1)::integer as idx,
    ean, category, format, language, pub_year,
    date_diff('day', DATE '{epoch}', cast(release_date as date)) as release_day,
    new_price
from read_parquet('{clean}/catalogue.parquet');
