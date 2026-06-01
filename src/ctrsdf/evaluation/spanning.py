from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def factor_alpha(strategy: pd.DataFrame, factors: pd.DataFrame, ret_col: str = "long_short") -> pd.Series:
    data = strategy.merge(factors, on="month_end", how="inner").dropna()
    factor_cols = [c for c in data.columns if c not in {"month_end", ret_col}]
    x = sm.add_constant(data[factor_cols], has_constant="add")
    fit = sm.OLS(data[ret_col], x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return pd.Series({"alpha": fit.params["const"], "alpha_t": fit.tvalues["const"], "n": fit.nobs})
