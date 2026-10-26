"""Tests for temporal isolation, recursive forecasts and physical inventory flow."""
import unittest
import numpy as np
import pandas as pd
from stockwise import predict_horizon, simulate_inventory


class AddOneModel:
    def predict(self, x):
        return x[:, 0] + 1


class ConstantModel:
    def predict(self, x):
        return np.full(len(x), 10.0)


class StockWiseTests(unittest.TestCase):
    def test_recursive_forecasts_feed_predictions_back(self):
        hist = np.zeros((1, 56))
        dates = pd.Series(pd.date_range("2020-01-01", periods=90))
        pred = predict_horizon(hist, dates, 56, 4, np.zeros((1, 3)), AddOneModel(), "poisson_boosting")
        np.testing.assert_array_equal(pred, [[1, 2, 3, 4]])
        np.testing.assert_array_equal(hist, np.zeros((1, 56)))

    def test_future_demand_cannot_change_previous_orders(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=84))
        original = np.ones((1, 84))
        changed = original.copy()
        changed[:, 70:] = 1000
        args = (dates, np.zeros((1, 3)), ["sku"], ConstantModel(), 56, 3)
        a = simulate_inventory(original, *args)
        b = simulate_inventory(changed, *args)
        mask = a.date < str(dates.iloc[70].date())
        np.testing.assert_array_equal(a.loc[mask, "ordered"], b.loc[mask, "ordered"])

    def test_receipts_wait_for_full_lead_time_and_stock_conserves(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=84))
        y = np.zeros((1, 84))
        y[:, 56:] = 5
        df = simulate_inventory(y, dates, np.zeros((1, 3)), ["sku"], ConstantModel(), 56, 3)
        df = df.query("method == 'poisson_boosting'").reset_index(drop=True)
        np.testing.assert_array_equal(df.fulfilled.iloc[:3], [0, 0, 0])
        self.assertEqual(df.fulfilled.iloc[3], 5)
        receipts = df.ordered.shift(3, fill_value=0)
        np.testing.assert_allclose(df.on_hand, receipts.cumsum() - df.fulfilled.cumsum())
        np.testing.assert_allclose(df.fulfilled + df.lost_sales, df.sales_proxy)


if __name__ == "__main__":
    unittest.main()
