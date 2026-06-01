-- FRB rates and yield-curve controls.
-- Replace {{ frb_rates_table }} with the schema-audit resolved table.
-- Parameters: %(start)s, %(end)s.

select
    date as month_end,
    gs3m,
    gs1,
    gs2,
    gs10,
    baa,
    aaa,
    fedfunds
from {{ frb_rates_table }}
where date between %(start)s and %(end)s;
