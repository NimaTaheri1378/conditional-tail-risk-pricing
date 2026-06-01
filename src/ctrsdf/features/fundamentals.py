from __future__ import annotations

import numpy as np
import pandas as pd


def build_annual_characteristics(funda: pd.DataFrame) -> pd.DataFrame:
    df = funda.copy()
    df.columns = [str(c).lower() for c in df.columns]
    required = {"gvkey", "datadate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Compustat funda missing required columns: {sorted(missing)}")
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["available_month"] = (df["datadate"] + pd.DateOffset(months=6)) + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({"gvkey": df["gvkey"], "available_month": df["available_month"]})
    if {"ceq", "txditc", "pstkrv"}.issubset(df.columns):
        out["book_equity"] = df["ceq"].fillna(0) + df["txditc"].fillna(0) - df["pstkrv"].fillna(0)
    if {"revt", "cogs", "at"}.issubset(df.columns):
        out["gross_profitability"] = (df["revt"] - df["cogs"]) / df["at"].replace(0, np.nan)
    if {"ib", "at"}.issubset(df.columns):
        out["operating_profitability"] = df["ib"] / df["at"].replace(0, np.nan)
    if "at" in df.columns:
        out["asset_growth"] = (
            df.sort_values(["gvkey", "datadate"])
            .groupby("gvkey")["at"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
        )
    if {"dltt", "dlc", "at"}.issubset(df.columns):
        out["leverage"] = (df["dltt"].fillna(0) + df["dlc"].fillna(0)) / df["at"].replace(0, np.nan)
    if {"che", "at"}.issubset(df.columns):
        out["cash_assets"] = df["che"] / df["at"].replace(0, np.nan)
    return out


def month_expand_characteristics(chars: pd.DataFrame, months: pd.Series) -> pd.DataFrame:
    calendar = pd.DataFrame({"month_end": pd.to_datetime(sorted(months.dropna().unique()))})
    rows = []
    for gvkey, group in chars.sort_values("available_month").groupby("gvkey"):
        expanded = pd.merge_asof(
            calendar,
            group.sort_values("available_month"),
            left_on="month_end",
            right_on="available_month",
            direction="backward",
        )
        expanded["gvkey"] = gvkey
        rows.append(expanded)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
