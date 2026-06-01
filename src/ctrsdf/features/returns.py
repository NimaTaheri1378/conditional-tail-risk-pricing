from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_crsp_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(c).lower() for c in out.columns]
    rename = {
        "dlyret": "ret",
        "retx": "ret_ex_div",
        "dlyprc": "prc",
        "dlycap": "mcap",
        "dlyvol": "vol",
        "permno": "permno",
    }
    return out.rename(columns={k: v for k, v in rename.items() if k in out.columns})


def daily_to_monthly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    df = normalize_crsp_columns(frame)
    if "month_end" in df.columns and "ret" in df.columns:
        df["month_end"] = pd.to_datetime(df["month_end"])
        df = df.sort_values(["permno", "month_end"]).copy()
        df["ret_fwd_1m"] = df.groupby("permno")["ret"].shift(-1)
        return df
    if "date" not in df or "permno" not in df:
        raise ValueError("CRSP daily data must contain date and permno.")
    df["date"] = pd.to_datetime(df["date"])
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
    if "ret" not in df:
        raise ValueError("CRSP daily data must contain ret or dlyret.")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
    grouped = df.sort_values(["permno", "date"]).groupby(["permno", "month_end"], as_index=False)
    out = grouped.agg(
        ret=("ret", lambda x: float(np.prod(1.0 + x) - 1.0)),
        trading_days=("ret", "size"),
        prc=("prc", "last") if "prc" in df else ("ret", "size"),
        mcap=("mcap", "last") if "mcap" in df else ("ret", "size"),
        vol=("vol", "sum") if "vol" in df else ("ret", "size"),
    )
    out["ret_fwd_1m"] = out.sort_values(["permno", "month_end"]).groupby("permno")["ret"].shift(-1)
    return out


def add_market_state(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    market = df.groupby("month_end", as_index=False).apply(
        lambda x: pd.Series({"mkt_ret": np.average(x["ret"], weights=x["mcap"].clip(lower=0))})
        if "mcap" in x and x["mcap"].fillna(0).sum() > 0
        else pd.Series({"mkt_ret": x["ret"].mean()})
    )
    return df.merge(market, on="month_end", how="left")
