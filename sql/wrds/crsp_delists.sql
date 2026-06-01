-- CRSP delisting audit extract.
-- Replace {{ crsp_delists_table }} with the schema-audit resolved table.
-- Parameters: %(start)s, %(end)s.

select
    permno,
    dlstdt,
    dlstcd,
    dlret,
    dlprc
from {{ crsp_delists_table }}
where dlstdt between %(start)s and %(end)s;
