import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1] / "01_car_price_prediction"
sys.path.insert(0, str(PROJECT / "src"))
from car_price import build_pipeline, train_model, predict_csv


def sample_cars(n=60, include_target=True):
    rng = np.random.default_rng(23)
    frame = pd.DataFrame({
        "model": rng.choice(["Fiesta", "Focus", "Kuga"], n),
        "year": rng.integers(2005, 2022, n),
        "transmission": rng.choice(["Manual", "Automatic"], n),
        "mileage": rng.integers(1000, 180000, n),
        "fuelType": rng.choice(["Petrol", "Diesel", "Hybrid"], n),
        "tax": rng.integers(0, 300, n),
        "mpg": rng.uniform(25, 75, n),
        "engineSize": rng.choice([1.0, 1.2, 1.5, 2.0], n),
    })
    if include_target:
        frame["price"] = (
            25000 + (frame["year"] - 2005) * 650
            - frame["mileage"] * 0.045
            + frame["engineSize"] * 1500
            + (frame["transmission"] == "Automatic") * 1200
        ).clip(lower=1000)
    return frame


class CarPriceTests(unittest.TestCase):
    def test_train_and_predict_with_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_path = root / "cars.csv"
            test_path = root / "new_cars.csv"
            model_path = root / "car_model.joblib"
            output_path = root / "predictions.csv"
            sample_cars().to_csv(train_path, index=False)
            sample_cars(8, include_target=False).to_csv(test_path, index=False)

            metrics = train_model(train_path, model_path, n_estimators=12)
            self.assertTrue(model_path.exists())
            self.assertIn("mae", metrics)

            result = predict_csv(test_path, output_path, model_path)
            self.assertEqual(len(result), 8)
            self.assertTrue((result["price"] > 0).all())

    def test_pipeline_handles_unseen_categories(self):
        frame = sample_cars(20)
        X = frame.drop(columns="price")
        model = build_pipeline(X, n_estimators=8)
        model.fit(X, np.log1p(frame["price"]))
        new = X.iloc[:2].copy()
        new.loc[new.index[0], "model"] = "BrandNewModel"
        predictions = model.predict(new)
        self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
