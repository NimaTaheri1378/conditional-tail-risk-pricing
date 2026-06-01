-- Compustat quarterly fundamentals reference query.
-- Replace {{ comp_fundq_table }} with the schema-audit resolved table.
-- Parameters: %(start)s, %(end)s.

select
    gvkey,
    datadate,
    fyearq,
    fqtr,
    rdq,
    atq,
    ceqq,
    saleq,
    ibq,
    niq,
    dlttq,
    dlcq,
    cheq,
    capxy,
    xrdq
from {{ comp_fundq_table }}
where datadate between %(start)s and %(end)s
  and indfmt = 'INDL'
  and consol = 'C'
  and datafmt = 'STD'
  and popsrc = 'D';
