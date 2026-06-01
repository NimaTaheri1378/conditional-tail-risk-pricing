from __future__ import annotations

import numpy as np
import pandas as pd


def long_short_decile_returns(
    frame: pd.DataFrame,
    score: str = "prediction",
    ret: str = "ret_fwd_1m",
    weight_col: str | None = "mcap",
    q: int = 10,
) -> pd.DataFrame:
    rows = []
    for date, group in frame.dropna(subset=[score, ret]).groupby("month_end"):
        if group[score].nunique() < q:
            continue
        sub = group.copy()
        sub["bucket"] = pd.qcut(sub[score], q, labels=False, duplicates="drop")
        low = sub[sub["bucket"] == sub["bucket"].min()]
        high = sub[sub["bucket"] == sub["bucket"].max()]
        long_ret = _weighted_mean(high, ret, weight_col)
        short_ret = _weighted_mean(low, ret, weight_col)
        rows.append({"month_end": date, "long": long_ret, "short": short_ret, "long_short": long_ret - short_ret})
    return pd.DataFrame(rows)


def _weighted_mean(frame: pd.DataFrame, value: str, weight: str | None) -> float:
    if weight and weight in frame and frame[weight].fillna(0).sum() > 0:
        w = frame[weight].clip(lower=0).fillna(0)
        return float(np.average(frame[value], weights=w))
    return float(frame[value].mean())
