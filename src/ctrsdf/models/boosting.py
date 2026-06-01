from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def fit_gradient_boosting(train: pd.DataFrame, target: str, features: list[str], use_gpu: bool = True):
    data = train.dropna(subset=[target])
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            objective="reg:squarederror",
            random_state=20260527,
        )
        model.fit(data[features], data[target])
        return model
    except Exception:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=50,
                        random_state=20260527,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        pipe.fit(data[features], data[target])
        return pipe
