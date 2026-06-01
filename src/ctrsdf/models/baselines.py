from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def numeric_features(frame: pd.DataFrame, target: str, exclude: set[str] | None = None) -> list[str]:
    exclude = (exclude or set()) | {target}
    return [
        c for c in frame.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(frame[c])
    ]


def fit_elastic_net(train: pd.DataFrame, target: str, features: list[str] | None = None):
    features = features or numeric_features(train, target, {"permno"})
    data = train.dropna(subset=[target])
    x = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=[0.001, 0.01, 0.1, 1.0])),
        ]
    )
    model.fit(x, data[target])
    return model, features


def fama_macbeth(frame: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for date, group in frame.dropna(subset=[target]).groupby("month_end"):
        sub = group[[target] + features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) <= len(features) + 5:
            continue
        x = sm.add_constant(sub[features].astype("float64"), has_constant="add")
        fit = sm.OLS(sub[target].astype("float64"), x).fit()
        row = {"month_end": date}
        row.update(fit.params.to_dict())
        rows.append(row)
    coefs = pd.DataFrame(rows)
    if coefs.empty:
        return coefs
    summary = coefs.drop(columns=["month_end"]).agg(["mean", "std", "count"]).T
    summary["tstat"] = summary["mean"] / (summary["std"] / np.sqrt(summary["count"]))
    return summary.reset_index(names="term")
