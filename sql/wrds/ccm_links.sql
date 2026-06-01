-- CRSP-Compustat link history.
-- Replace {{ ccm_links_table }} with the schema-audit resolved table.
-- Parameters: %(start)s, %(end)s.

select
    gvkey,
    lpermno::integer as permno,
    linkdt,
    linkenddt,
    linktype,
    linkprim
from {{ ccm_links_table }}
where coalesce(linkenddt, '2100-01-01') >= %(start)s
  and linkdt <= %(end)s
  and linktype in ('LU', 'LC', 'LS')
  and linkprim in ('P', 'C')
  and lpermno is not null;
