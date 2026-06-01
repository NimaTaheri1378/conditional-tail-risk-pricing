from __future__ import annotations

from pathlib import Path

import pandas as pd

from ctrsdf.config import ProjectConfig
from ctrsdf.features.fundamentals import build_annual_characteristics
from ctrsdf.features.options import build_option_features, add_option_interactions
from ctrsdf.features.returns import daily_to_monthly_returns
from ctrsdf.utils.io import atomic_write_parquet
from ctrsdf.utils.manifest import Manifest


def _read_parquet_path(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_feature_store(config: ProjectConfig, label: str) -> Path:
    raw = config.path("data_raw") / label
    processed = config.path("data_processed") / label
    processed.mkdir(parents=True, exist_ok=True)

    crsp_path = raw / "crsp_monthly"
    crsp = _read_parquet_path(crsp_path)
    monthly = daily_to_monthly_returns(crsp)
    panel = monthly

    ff_path = raw / "ff_factors.parquet"
    if ff_path.exists():
        ff = pd.read_parquet(ff_path)
        ff["month_end"] = pd.to_datetime(ff["month_end"]) + pd.offsets.MonthEnd(0)
        ff = ff.sort_values("month_end")
        ff["rf_fwd_1m"] = ff["rf"].shift(-1)
        panel = panel.merge(ff[["month_end", "mktrf", "smb", "hml", "rf", "umd", "rf_fwd_1m"]], on="month_end", how="left")
        panel["ret_excess_fwd_1m"] = panel["ret_fwd_1m"] - panel["rf_fwd_1m"]

    ccm_path = raw / "ccm_links.parquet"
    if ccm_path.exists():
        panel = _attach_ccm_links(panel, pd.read_parquet(ccm_path))

    comp_path = raw / "comp_funda.parquet"
    if comp_path.exists() and "gvkey" in panel.columns:
        chars = build_annual_characteristics(pd.read_parquet(comp_path))
        if not chars.empty:
            panel = _attach_fundamentals(panel, chars)

    option_path = raw / "option_surface.parquet"
    if not option_path.exists():
        option_path = raw / "option_surface"
    link_path = raw / "crsp_optionm_link.parquet"
    if option_path.exists() and link_path.exists():
        opts = build_option_features(_read_parquet_path(option_path))
        opts = _attach_option_links(opts, pd.read_parquet(link_path))
        panel = panel.merge(opts, on=["permno", "month_end"], how="left")

    vix_path = raw / "cboe_vix.parquet"
    if vix_path.exists():
        vix = pd.read_parquet(vix_path)
        vix["month_end"] = pd.to_datetime(vix["month_end"])
        panel = panel.merge(vix, on="month_end", how="left")

    rates_path = raw / "frb_rates.parquet"
    if rates_path.exists():
        rates = pd.read_parquet(rates_path)
        rates["month_end"] = pd.to_datetime(rates["month_end"]) + pd.offsets.MonthEnd(0)
        if {"gs10", "gs3m"}.issubset(rates.columns):
            rates["treasury_slope"] = rates["gs10"] - rates["gs3m"]
        panel = panel.merge(rates, on="month_end", how="left")

    panel = add_option_interactions(panel)
    numeric_cols = panel.select_dtypes(include="number").columns
    panel[numeric_cols] = panel[numeric_cols].replace([float("inf"), float("-inf")], float("nan"))
    output = processed / "monthly_feature_store.parquet"
    atomic_write_parquet(panel, output)
    Manifest(
        name=f"{label}_feature_store",
        status="completed",
        outputs={"path": str(output), "rows": int(len(panel)), "columns": list(panel.columns)},
    ).write(config.path("manifests") / f"{label}_feature_store.json")
    return output


def _attach_ccm_links(panel: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    links = links.copy()
    links["linkdt"] = pd.to_datetime(links["linkdt"])
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(pd.Timestamp("2100-01-01"))
    merged = df.merge(links, on="permno", how="left")
    keep = merged["gvkey"].isna() | (
        (merged["month_end"] >= merged["linkdt"]) & (merged["month_end"] <= merged["linkenddt"])
    )
    merged = merged.loc[keep].sort_values(["permno", "month_end", "linkprim"])
    return merged.drop_duplicates(["permno", "month_end"], keep="first")


def _attach_fundamentals(panel: pd.DataFrame, chars: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    chars = chars.copy()
    chars["available_month"] = pd.to_datetime(chars["available_month"])
    out = df.merge(chars, on="gvkey", how="left")
    out = out[out["available_month"].isna() | (out["available_month"] <= out["month_end"])]
    out = out.sort_values(["permno", "month_end", "available_month"])
    return out.drop_duplicates(["permno", "month_end"], keep="last")


def _attach_option_links(options: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    opts = options.copy()
    links = links.copy()
    links["sdate"] = pd.to_datetime(links["sdate"])
    links["edate"] = pd.to_datetime(links["edate"]).fillna(pd.Timestamp("2100-01-01"))
    merged = opts.merge(links, on="secid", how="left")
    merged = merged[
        merged["permno"].notna()
        & (merged["month_end"] >= merged["sdate"])
        & (merged["month_end"] <= merged["edate"])
    ]
    merged = merged.sort_values(["permno", "month_end", "score"])
    merged = merged.drop_duplicates(["permno", "month_end"], keep="first")
    return merged.drop(columns=["sdate", "edate", "score", "secid"])
