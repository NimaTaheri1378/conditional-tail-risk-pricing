from __future__ import annotations

import numpy as np
import pandas as pd


def torch_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class NeuralSDF:
    def __init__(self, input_dim: int, device: str = "cuda"):
        import torch
        from torch import nn

        self.torch = torch
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ).to(self.device)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 25, batch_size: int = 8192):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.x_mean = np.nanmean(x, axis=0)
        self.x_std = np.nanstd(x, axis=0)
        self.x_std[self.x_std == 0] = 1.0
        self.y_mean = float(np.nanmean(y))
        self.y_std = float(np.nanstd(y))
        if self.y_std == 0:
            self.y_std = 1.0
        x_scaled = (x - self.x_mean) / self.x_std
        y_scaled = (y - self.y_mean) / self.y_std
        dataset = TensorDataset(
            torch.tensor(x_scaled, dtype=torch.float32),
            torch.tensor(y_scaled.reshape(-1, 1), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        self.net.train()
        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        x_scaled = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            pred = self.net(torch.tensor(x_scaled, dtype=torch.float32).to(self.device)).cpu().numpy()
        return pred.ravel() * self.y_std + self.y_mean


def fit_neural_sdf(train: pd.DataFrame, target: str, features: list[str], epochs: int = 25):
    data = train.dropna(subset=[target]).copy()
    raw = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = raw.median().fillna(0.0)
    x = raw.fillna(med).fillna(0.0).astype("float64").to_numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = data[target].astype("float64").to_numpy()
    model = NeuralSDF(input_dim=x.shape[1])
    model.fit(x, y, epochs=epochs)
    return model
