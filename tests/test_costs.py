import pandas as pd

from ctrsdf.portfolio.costs import transaction_cost


def test_transaction_cost_nonnegative():
    costs = transaction_cost(pd.Series([0.001]), pd.Series([0.2]), pd.Series([0.01]), 0.1)
    assert costs.iloc[0] > 0
