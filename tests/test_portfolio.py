import pandas as pd

from ctrsdf.portfolio.backtest import long_short_decile_returns


def test_long_short_decile_returns():
    frame = pd.DataFrame(
        {
            "month_end": ["2020-01-31"] * 20,
            "prediction": range(20),
            "ret_fwd_1m": [0.0] * 10 + [0.1] * 10,
            "mcap": [1.0] * 20,
        }
    )
    out = long_short_decile_returns(frame, q=10)
    assert out.iloc[0]["long_short"] > 0
