from __future__ import annotations

import pandas as pd


def build_option_features(surface: pd.DataFrame) -> pd.DataFrame:
    df = surface.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "month_end" not in df or "secid" not in df:
        raise ValueError("OptionMetrics surface data must contain month_end and secid.")
    df["month_end"] = pd.to_datetime(df["month_end"])
    pivot = df.pivot_table(
        index=["secid", "month_end"],
        columns=["days", "delta", "cp_flag"],
        values="impl_volatility",
        aggfunc="median",
    )

    def col(days: int, delta: int, cp: str) -> pd.Series:
        key = (float(days), float(delta), cp)
        return pivot[key] if key in pivot else pd.Series(index=pivot.index, dtype="float64")

    out = pd.DataFrame(index=pivot.index)
    out["atm_iv_30"] = col(30, 50, "C")
    out["atm_iv_60"] = col(60, 50, "C")
    out["atm_iv_91"] = col(91, 50, "C")
    out["put25_iv_30"] = col(30, -25, "P")
    out["call25_iv_30"] = col(30, 25, "C")
    out["put10_iv_30"] = col(30, -10, "P")
    out["skew_25d"] = out["put25_iv_30"] - out["call25_iv_30"]
    out["tail_slope_10p_25p"] = out["put10_iv_30"] - out["put25_iv_30"]
    out["term_slope"] = out["atm_iv_91"] - out["atm_iv_30"]
    out["curvature"] = (out["put25_iv_30"] + out["call25_iv_30"]) / 2.0 - out["atm_iv_30"]
    return out.reset_index()


def add_option_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    pairs = [
        ("skew_25d", "vix_level", "skew_25d_x_vix_level"),
        ("skew_25d", "leverage", "skew_25d_x_leverage"),
        ("atm_iv_30", "cash_assets", "atm_iv_30_x_cash_assets"),
        ("tail_slope_10p_25p", "asset_growth", "tail_slope_10p_25p_x_asset_growth"),
        ("atm_iv_30", "market_drawdown", "atm_iv_30_x_market_drawdown"),
    ]
    for left, right, name in pairs:
        if left in df and right in df:
            df[name] = df[left] * df[right]
    return df
