-- Compustat annual fundamentals.
-- Replace {{ comp_funda_table }} with the schema-audit resolved table.
-- Parameters: %(start)s, %(end)s.

select
    gvkey,
    datadate,
    fyear,
    at,
    ceq,
    txditc,
    pstkrv,
    revt,
    cogs,
    ib,
    dltt,
    dlc,
    che,
    capx,
    xrd,
    sale,
    ni,
    oancf,
    prstkc,
    sstk,
    seq,
    pstk
from {{ comp_funda_table }}
where datadate between %(start)s and %(end)s
  and indfmt = 'INDL'
  and consol = 'C'
  and datafmt = 'STD'
  and popsrc = 'D';
