import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.house_price import (
    build_model, calculate_metrics, create_submission, load_training_data,
    predict_frame, train_model,
)


def sample_houses(n=70, include_target=True):
    rng = np.random.default_rng(17)
    data = pd.DataFrame({
        "Id": np.arange(1, n + 1),
        "OverallQual": rng.integers(2, 10, n),
        "GrLivArea": rng.integers(500, 3000, n),
        "YearBuilt": rng.integers(1950, 2020, n),
        "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown"], n),
        "GarageType": rng.choice(["Attchd", "Detchd", None], n),
    })
    if include_target:
        data["SalePrice"] = (
            30000 + data["OverallQual"] * 15000 + data["GrLivArea"] * 65
            + (data["YearBuilt"] - 1950) * 350
        ).astype(float)
    return data


class HousePriceTests(unittest.TestCase):
    def test_pipeline_fits_and_predicts_positive_prices(self):
        data = sample_houses()
        X = data.drop(columns=["SalePrice", "Id"])
        y = np.log1p(data["SalePrice"])
        model = build_model(n_estimators=15, max_depth=2)
        model.fit(X, y)
        predictions = np.expm1(model.predict(X.iloc[:4]))
        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.all(predictions > 0))

    def test_train_predict_and_submission_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = sample_houses(80)
            test = sample_houses(12, include_target=False)
            train_path = root / "train.csv"
            test_path = root / "test.csv"
            model_path = root / "model.joblib"
            metrics_path = root / "metrics.json"
            output_path = root / "submission.csv"
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            result = train_model(
                train_path, model_path, metrics_path,
                n_estimators=15, max_depth=2, validation_size=0.2,
            )
            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertGreaterEqual(result["metrics"]["r2"], -1.0)

            submission = create_submission(test_path, output_path, model_path)
            self.assertEqual(len(submission), len(test))
            self.assertEqual(list(submission.columns), ["Id", "SalePrice"])
            self.assertTrue((submission["SalePrice"] > 0).all())

    def test_missing_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            pd.DataFrame({"Id": [1], "price": [200000]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_training_data(path)

    def test_metrics(self):
        metrics = calculate_metrics([100, 200, 300], np.array([110, 190, 310]))
        self.assertIn("mae", metrics)
        self.assertGreater(metrics["rmse"], 0)


if __name__ == "__main__":
    unittest.main()
