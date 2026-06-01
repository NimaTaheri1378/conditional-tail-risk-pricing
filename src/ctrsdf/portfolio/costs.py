from __future__ import annotations

import numpy as np
import pandas as pd


def transaction_cost(half_spread: pd.Series, volatility: pd.Series, trade_fraction_adv: pd.Series, eta: float) -> pd.Series:
    return half_spread.fillna(0) + eta * volatility.fillna(0) * np.sqrt(trade_fraction_adv.clip(lower=0).fillna(0))
