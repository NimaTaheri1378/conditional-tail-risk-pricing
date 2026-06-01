-- Reduced raw-chain validation sample.
-- Replace {{ option_price_table }} with the year-specific OptionMetrics price table.
-- Parameters: %(start)s, %(end)s.

with sample_date as (
  select max(date) as date
  from {{ option_price_table }}
  where date between %(start)s and %(end)s
)
select
    o.secid,
    o.date,
    o.cp_flag,
    o.delta,
    o.impl_volatility,
    o.best_bid,
    o.best_offer,
    o.volume,
    o.open_interest,
    o.exdate,
    o.strike_price
from {{ option_price_table }} o
join sample_date s on o.date = s.date
where o.best_bid > 0
  and o.best_offer > o.best_bid
  and o.impl_volatility between 0.01 and 3.0
  and o.delta is not null
  and o.exdate between o.date + interval '20 day' and o.date + interval '220 day';
