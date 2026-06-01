-- Fama-French factors and public benchmark test assets.
-- Replace {{ ff_factors_table }} and {{ ff_test_portfolios_table }} with schema-audit resolved tables.
-- Parameters: %(start)s, %(end)s.

select date as month_end, mktrf, smb, hml, rf, umd
from {{ ff_factors_table }}
where date between %(start)s and %(end)s;

select *
from {{ ff_test_portfolios_table }}
where date between %(start)s and %(end)s;
