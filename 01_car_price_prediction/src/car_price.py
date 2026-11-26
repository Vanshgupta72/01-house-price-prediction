from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

TARGET = "price"


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Ford dataset not found: {path}\n"
            "Download the correctly licensed Ford used-car dataset and save it as data/ford.csv."
        )
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' target column.")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[TARGET]).copy()
    frame = frame.loc[frame[TARGET] > 0]
    if frame.empty:
        raise ValueError("Dataset has no valid positive car prices.")
    return frame


def build_pipeline(X: pd.DataFrame, *, n_estimators: int = 350, random_state: int = 42):
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()
    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable feature columns found.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train_model(data_path, model_path, metrics_path=None, *, n_estimators=350, random_state=42):
    frame = load_data(data_path)
    X = frame.drop(columns=[TARGET]).copy()
    # The original Ford dataset commonly includes an unnamed index column.
    X = X.loc[:, ~X.columns.astype(str).str.match(r"^Unnamed:")]
    if X.empty:
        raise ValueError("No feature columns remain after removing the target/index.")
    y = frame[TARGET].astype(float)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    validation_model = build_pipeline(X_train, n_estimators=n_estimators, random_state=random_state)
    validation_model.fit(X_train, np.log1p(y_train))
    validation_predictions = np.maximum(np.expm1(validation_model.predict(X_valid)), 0.0)

    metrics = {
        "mae": float(mean_absolute_error(y_valid, validation_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_valid, validation_predictions))),
        "r2": float(r2_score(y_valid, validation_predictions)),
    }

    final_model = build_pipeline(X, n_estimators=n_estimators, random_state=random_state)
    final_model.fit(X, np.log1p(y))
    bundle: dict[str, Any] = {
        "model": final_model,
        "target": TARGET,
        "target_transform": "log1p",
        "feature_columns": list(X.columns),
        "training_rows": int(len(X)),
        "validation_metrics_original_price_scale": metrics,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    if metrics_path:
        import json
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def predict_csv(input_path, output_path, model_path):
    input_path, output_path, model_path = map(Path, (input_path, output_path, model_path))
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first.")
    data = pd.read_csv(input_path)
    bundle = joblib.load(model_path)
    expected = list(bundle["feature_columns"])
    missing = [column for column in expected if column not in data.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing[:10]}")
    features = data.loc[:, expected]
    predictions = np.maximum(np.expm1(bundle["model"].predict(features)), 0.0)
    result = pd.DataFrame({"price": predictions})
    if "id" in data.columns:
        result.insert(0, "id", data["id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
