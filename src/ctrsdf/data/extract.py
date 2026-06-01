from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ctrsdf.config import ProjectConfig
from ctrsdf.data.schema import load_schema_audit
from ctrsdf.data.wrds_conn import raw_sql
from ctrsdf.utils.io import atomic_write_parquet
from ctrsdf.utils.manifest import Manifest, sha256_text

LOGGER = logging.getLogger(__name__)


def _resolved_map(config: ProjectConfig) -> dict[str, str]:
    audit = load_schema_audit(config)
    return {
        row.logical_name: f"{row.library}.{row.table}"
        for row in audit.itertuples()
        if bool(row.resolved)
    }


def _split_table(table: str) -> tuple[str, str]:
    library, name = table.split(".", 1)
    return library, name


def _year_ranges(start: str, end: str) -> list[tuple[int, str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    ranges = []
    for year in range(start_ts.year, end_ts.year + 1):
        left = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        right = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        if left <= right:
            ranges.append((year, left.strftime("%Y-%m-%d"), right.strftime("%Y-%m-%d")))
    return ranges


def smoke_extract(config: ProjectConfig) -> dict[str, Path]:
    return extract_all(config, config.smoke_start, config.smoke_end, label="smoke")


def full_extract(config: ProjectConfig) -> dict[str, Path]:
    return extract_all(config, config.sample_start, config.sample_end, label="full")


def extract_all(config: ProjectConfig, start: str, end: str, label: str) -> dict[str, Path]:
    tables = _resolved_map(config)
    outputs: dict[str, Path] = {}

    if "crsp_daily" in tables and "crsp_names" in tables:
        outputs["crsp_monthly"] = extract_crsp_monthly_from_daily(
            config, tables["crsp_daily"], tables["crsp_names"], start, end, label
        )
    if "comp_funda" in tables:
        outputs["comp_funda"] = extract_comp_funda(config, tables["comp_funda"], start, end, label)
    if "ccm_links" in tables:
        outputs["ccm_links"] = extract_ccm_links(config, tables["ccm_links"], start, end, label)
    if "option_surface" in tables:
        outputs["option_surface"] = extract_option_surface_features(
            config, tables["option_surface"], start, end, label
        )
    if "crsp_optionm_link" in tables:
        outputs["crsp_optionm_link"] = extract_crsp_optionm_link(
            config, tables["crsp_optionm_link"], start, end, label
        )
    if "cboe_vix" in tables:
        outputs["cboe_vix"] = extract_vix(config, tables["cboe_vix"], start, end, label)
    if "ff_factors" in tables:
        outputs["ff_factors"] = extract_ff_factors(config, tables["ff_factors"], start, end, label)
    if "ff_test_portfolios" in tables:
        outputs["ff_test_portfolios"] = extract_ff_test_portfolios(
            config, tables["ff_test_portfolios"], start, end, label
        )
    if "frb_rates" in tables:
        outputs["frb_rates"] = extract_frb_rates(config, tables["frb_rates"], start, end, label)

    Manifest(
        name=f"{label}_extract",
        status="completed",
        parameters={"start": start, "end": end},
        outputs={key: str(path) for key, path in outputs.items()},
    ).write(config.path("manifests") / f"{label}_extract.json")
    return outputs


def _query_to_parquet(
    config: ProjectConfig,
    query: str,
    path: Path,
    manifest_name: str,
    params: dict | None = None,
) -> Path:
    if path.exists():
        LOGGER.info("Skipping existing extract %s", path)
        return path
    frame = raw_sql(query, params=params)
    atomic_write_parquet(frame, path)
    Manifest(
        name=manifest_name,
        status="completed",
        parameters=params or {},
        outputs={"path": str(path), "rows": int(len(frame))},
        diagnostics={"columns": list(frame.columns), "sql_hash": sha256_text(query)},
    ).write(config.path("manifests") / f"{manifest_name}.json")
    LOGGER.info("Wrote %s rows to %s", len(frame), path)
    return path


def extract_crsp_monthly_from_daily(
    config: ProjectConfig,
    daily_table: str,
    names_table: str,
    start: str,
    end: str,
    label: str,
) -> Path:
    outdir = config.path("data_raw") / label / "crsp_monthly"
    outdir.mkdir(parents=True, exist_ok=True)
    shard_rows = {}
    for year, left, right in _year_ranges(start, end):
        path = outdir / f"year={year}.parquet"
        query = f"""
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
            from {daily_table} d
            join {names_table} n
              on d.permno = n.permno
             and d.dlycaldt between n.secinfostartdt and n.secinfoenddt
            where d.dlycaldt between %(start)s and %(end)s
              and n.usincflg = 'Y'
              and n.securitytype = 'EQTY'
              and n.securitysubtype = 'COM'
              and n.sharetype = 'NS'
              and n.primaryexch in ('N', 'A', 'Q')
              and n.tradingstatusflg = 'A'
            group by d.permno, month_end
        """
        _query_to_parquet(config, query, path, f"{label}_crsp_monthly_{year}", {"start": left, "end": right})
        shard_rows[str(year)] = int(pd.read_parquet(path, columns=["permno"]).shape[0])
    Manifest(
        name=f"{label}_crsp_monthly",
        status="completed",
        parameters={"start": start, "end": end},
        outputs={"path": str(outdir), "shards": shard_rows},
    ).write(config.path("manifests") / f"{label}_crsp_monthly.json")
    return outdir


def extract_comp_funda(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select gvkey, datadate, fyear, at, ceq, txditc, pstkrv, revt, cogs, ib,
               dltt, dlc, che, capx, xrd, sale, ni, oancf, prstkc, sstk, seq, pstk
        from {table}
        where datadate between %(start)s and %(end)s
          and indfmt = 'INDL'
          and consol = 'C'
          and datafmt = 'STD'
          and popsrc = 'D'
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "comp_funda.parquet",
        f"{label}_comp_funda",
        {"start": start, "end": end},
    )


def extract_ccm_links(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select gvkey, lpermno::integer as permno, linkdt, linkenddt, linktype, linkprim
        from {table}
        where coalesce(linkenddt, '2100-01-01') >= %(start)s
          and linkdt <= %(end)s
          and linktype in ('LU', 'LC', 'LS')
          and linkprim in ('P', 'C')
          and lpermno is not null
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "ccm_links.parquet",
        f"{label}_ccm_links",
        {"start": start, "end": end},
    )


def extract_crsp_optionm_link(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select secid, permno, sdate, edate, score
        from {table}
        where coalesce(edate, '2100-01-01') >= %(start)s
          and sdate <= %(end)s
          and permno is not null
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "crsp_optionm_link.parquet",
        f"{label}_crsp_optionm_link",
        {"start": start, "end": end},
    )


def extract_option_surface_features(config: ProjectConfig, resolved_table: str, start: str, end: str, label: str) -> Path:
    library, first_table = _split_table(resolved_table)
    prefix = "".join(ch for ch in first_table if not ch.isdigit())
    outdir = config.path("data_raw") / label / "option_surface"
    outdir.mkdir(parents=True, exist_ok=True)
    shard_rows = {}
    for year, left, right in _year_ranges(start, end):
        table = f"{library}.{prefix}{year}"
        path = outdir / f"year={year}.parquet"
        query = f"""
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
              from {table}
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
            group by secid, month_end, days, delta, cp_flag
        """
        try:
            _query_to_parquet(config, query, path, f"{label}_option_surface_{year}", {"start": left, "end": right})
            shard_rows[str(year)] = int(pd.read_parquet(path, columns=["secid"]).shape[0])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping option surface %s after %s", table, type(exc).__name__)
            shard_rows[str(year)] = -1
    Manifest(
        name=f"{label}_option_surface",
        status="completed",
        parameters={"start": start, "end": end},
        outputs={"path": str(outdir), "shards": shard_rows},
    ).write(config.path("manifests") / f"{label}_option_surface.json")
    return outdir


def extract_vix(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select
            (date_trunc('month', date) + interval '1 month - 1 day')::date as month_end,
            avg(vix) as vix_level,
            max(vix) as vix_max
        from {table}
        where date between %(start)s and %(end)s
        group by month_end
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "cboe_vix.parquet",
        f"{label}_cboe_vix",
        {"start": start, "end": end},
    )


def extract_ff_factors(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select date as month_end, mktrf, smb, hml, rf, umd
        from {table}
        where date between %(start)s and %(end)s
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "ff_factors.parquet",
        f"{label}_ff_factors",
        {"start": start, "end": end},
    )


def extract_ff_test_portfolios(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select *
        from {table}
        where date between %(start)s and %(end)s
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "ff_test_portfolios.parquet",
        f"{label}_ff_test_portfolios",
        {"start": start, "end": end},
    )


def extract_frb_rates(config: ProjectConfig, table: str, start: str, end: str, label: str) -> Path:
    query = f"""
        select date as month_end, gs3m, gs1, gs2, gs10, baa, aaa, fedfunds
        from {table}
        where date between %(start)s and %(end)s
    """
    return _query_to_parquet(
        config,
        query,
        config.path("data_raw") / label / "frb_rates.parquet",
        f"{label}_frb_rates",
        {"start": start, "end": end},
    )
