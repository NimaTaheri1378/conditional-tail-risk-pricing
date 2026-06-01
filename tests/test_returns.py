import pandas as pd

from ctrsdf.features.returns import daily_to_monthly_returns


def test_daily_to_monthly_compounds_returns():
    frame = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "date": ["2020-01-02", "2020-01-03", "2020-02-03"],
            "ret": [0.10, -0.10, 0.05],
            "prc": [10, 9, 10],
            "mcap": [100, 90, 110],
            "vol": [10, 20, 30],
        }
    )
    out = daily_to_monthly_returns(frame)
    jan = out[out["month_end"] == pd.Timestamp("2020-01-31")].iloc[0]
    assert round(jan["ret"], 6) == -0.01
    assert round(jan["ret_fwd_1m"], 6) == 0.05
