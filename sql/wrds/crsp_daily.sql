-- CRSP daily-to-monthly rebuild.
-- Replace {{ crsp_daily_table }} and {{ crsp_names_table }} with the schema-audit resolved tables.
-- Parameters: %(start)s, %(end)s.

select
    d.permno,
    (date_trunc('month', d.dlycaldt) + interval '1 month - 1 day')::date as month_end,
    case
      when min(d.dlyret) <= -1 then -1.0
      else exp(sum(case when d.dlyret > -1 then ln(1.0 + d.dlyret) else 0 end)) - 1.0
    end as ret,
    count(*) as trading_days,
    (array_agg(d.dlyprc order by d.dlycaldt desc))[1] as prc,
    (array_agg(d.dlycap order by d.dlycaldt desc))[1] as mcap,
    sum(d.dlyvol) as vol,
    avg(abs(d.dlyprc) * d.dlyvol) as adv_dollar,
    avg((d.dlyask - d.dlybid) / nullif((d.dlyask + d.dlybid) / 2.0, 0)) as quoted_spread
from {{ crsp_daily_table }} d
join {{ crsp_names_table }} n
  on d.permno = n.permno
 and d.dlycaldt between n.secinfostartdt and n.secinfoenddt
where d.dlycaldt between %(start)s and %(end)s
  and n.usincflg = 'Y'
  and n.securitytype = 'EQTY'
  and n.securitysubtype = 'COM'
  and n.sharetype = 'NS'
  and n.primaryexch in ('N', 'A', 'Q')
  and n.tradingstatusflg = 'A'
group by d.permno, month_end;
