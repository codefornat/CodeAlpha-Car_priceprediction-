import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from car_price import FeatureEngineer, RAW_FEATURES, build_pipeline


def sample():
    return pd.DataFrame([
        [" Swift ", 2015, 6.87, 30000, "Petrol", "Dealer", "Manual", 0],
        ["city", 2018, 12.0, 500, "Petrol", "Individual", "Automatic", 1],
        ["i20", 2014, 8.0, 40000, "Diesel", "Dealer", "Manual", 0],
    ], columns=RAW_FEATURES)


class WorkflowTests(unittest.TestCase):
    def test_age_usage_and_brand(self):
        result = FeatureEngineer().fit_transform(sample())
        self.assertEqual(result.loc[0, "Brand"], "Maruti Suzuki")
        self.assertEqual(result.loc[0, "Car_Age"], 3)
        self.assertEqual(result.loc[0, "Km_Per_Year"], 10000)
        self.assertEqual(result.loc[1, "Km_Per_Year"], 500)
        self.assertNotIn("Selling_Price", result)

    def test_missing_and_unknown_values(self):
        model = build_pipeline(RandomForestRegressor(n_estimators=5, random_state=42))
        model.fit(sample(), [4, 9, 5])
        data = sample().iloc[[0]].copy()
        data.loc[0, ["Present_Price", "Driven_kms", "Fuel_Type"]] = np.nan
        data.loc[0, "Car_Name"] = "unseen model"
        self.assertTrue(np.isfinite(model.predict(data)).all())

    def test_invalid_inputs_rejected(self):
        for col, value in [("Year", 2026), ("Driven_kms", -1), ("Owner", 0.5), ("Present_Price", "bad")]:
            data = sample().astype(object)
            data.loc[0, col] = value
            with self.assertRaises(ValueError):
                FeatureEngineer().fit_transform(data)
        with self.assertRaises(ValueError):
            FeatureEngineer().fit_transform(sample().drop(columns="Year"))


if __name__ == "__main__":
    unittest.main()
