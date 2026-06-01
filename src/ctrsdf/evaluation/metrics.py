from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_metrics(frame: pd.DataFrame, y: str, yhat: str) -> dict[str, float]:
    sub = frame[[y, yhat]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return {"n": 0.0}
    err = sub[yhat] - sub[y]
    denom = ((sub[y] - sub[y].mean()) ** 2).sum()
    r2 = 1.0 - float((err**2).sum() / denom) if denom else np.nan
    return {
        "n": float(len(sub)),
        "mse": float((err**2).mean()),
        "mae": float(err.abs().mean()),
        "oos_r2": r2,
        "spearman_ic": float(sub[y].corr(sub[yhat], method="spearman")),
    }


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {"n": 0.0}
    ann_mean = 12.0 * r.mean()
    ann_vol = np.sqrt(12.0) * r.std()
    wealth = (1.0 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "n": float(len(r)),
        "ann_mean": float(ann_mean),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_mean / ann_vol) if ann_vol else np.nan,
        "max_drawdown": float(drawdown.min()),
    }
