-- OptionMetrics standardized surface features.
-- Replace {{ option_surface_table }} with the year-specific schema-audit resolved table.
-- Parameters: %(start)s, %(end)s.

with ranked as (
  select
      secid,
      date,
      (date_trunc('month', date) + interval '1 month - 1 day')::date as month_end,
      days,
      delta,
      cp_flag,
      impl_volatility,
      row_number() over (
        partition by secid, date_trunc('month', date), days, delta, cp_flag
        order by date desc
      ) as rn
  from {{ option_surface_table }}
  where date between %(start)s and %(end)s
    and days in (30, 60, 91, 182)
    and delta in (-50, -25, -10, 25, 50)
    and impl_volatility is not null
)
select
    secid,
    month_end,
    days,
    delta,
    cp_flag,
    percentile_cont(0.5) within group (order by impl_volatility) as impl_volatility
from ranked
where rn <= 5
group by secid, month_end, days, delta, cp_flag;
