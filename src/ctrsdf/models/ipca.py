from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


class SimpleIPCA:
    """Lightweight IPCA-style characteristic-managed factor approximation."""

    def __init__(self, n_factors: int = 5, alpha: float = 0.001):
        self.n_factors = n_factors
        self.alpha = alpha
        self.pca = PCA(n_components=n_factors, random_state=20260527)
        self.reg = Ridge(alpha=alpha)
        self.features: list[str] = []

    def fit(self, frame: pd.DataFrame, features: list[str], target: str):
        data = frame.dropna(subset=[target]).copy()
        raw = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = raw.median().fillna(0.0)
        x = raw.fillna(med).fillna(0.0).astype("float64").to_numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = self.pca.fit_transform(x)
        self.reg.fit(z, data[target])
        self.features = features
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = (
            frame[self.features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float64")
            .to_numpy()
        )
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.reg.predict(self.pca.transform(x))
