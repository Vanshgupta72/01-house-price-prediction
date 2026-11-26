from __future__ import annotations

import json
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
from xgboost import XGBRegressor

TARGET = "SalePrice"
ID_COLUMN = "Id"


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the Ames-style training CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    frame = pd.read_csv(path)
    required = {TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Training data is missing required column(s): {sorted(missing)}")
    if frame.empty:
        raise ValueError("Training data is empty.")
    frame = frame.loc[frame[TARGET].notna()].copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.loc[frame[TARGET].notna() & (frame[TARGET] > 0)].copy()
    if frame.empty:
        raise ValueError("No rows with a valid positive SalePrice were found.")
    return frame


def split_features_target(frame: pd.DataFrame):
    """Separate target and remove the identifier from model features."""
    if TARGET not in frame.columns:
        raise ValueError(f"Expected target column '{TARGET}'.")
    y = frame[TARGET].astype(float)
    X = frame.drop(columns=[TARGET], errors="ignore")
    X = X.drop(columns=[ID_COLUMN], errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after removing the target and ID.")
    return X, y


def build_model(
    *,
    n_estimators: int = 500,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-safe preprocessing + XGBoost regression pipeline."""
    numeric_features = None
    categorical_features = None

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector_numeric()),
            ("categorical", categorical_pipeline, make_column_selector_categorical()),
        ],
        remainder="drop",
    )

    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    return Pipeline([("preprocessor", preprocessor), ("model", regressor)])


def make_column_selector_numeric():
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_include=np.number)


def make_column_selector_categorical():
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_exclude=np.number)


def predict_original_scale(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predict log1p(SalePrice), then invert the transform to dollar prices."""
    log_predictions = np.asarray(model.predict(features), dtype=float)
    prices = np.expm1(log_predictions)
    return np.maximum(prices, 0.0)


def calculate_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(actual_values, predicted_values)))
    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": rmse,
        "r2": float(r2_score(actual_values, predicted_values)),
    }


def train_model(
    data_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path | None = None,
    *,
    n_estimators: int = 500,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    random_state: int = 42,
    validation_size: float = 0.2,
) -> dict[str, Any]:
    """Evaluate on a holdout split, then refit on all rows and persist the model bundle."""
    frame = load_training_data(data_path)
    X, y = split_features_target(frame)
    if not 0.05 <= validation_size <= 0.5:
        raise ValueError("validation_size must be between 0.05 and 0.5.")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=validation_size, random_state=random_state
    )

    evaluation_model = build_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    evaluation_model.fit(X_train, np.log1p(y_train))
    validation_predictions = predict_original_scale(evaluation_model, X_valid)
    metrics = calculate_metrics(y_valid, validation_predictions)

    # Refit the final estimator on all available labeled records after evaluation.
    final_model = build_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    final_model.fit(X, np.log1p(y))

    bundle = {
        "model": final_model,
        "target": TARGET,
        "target_transform": "log1p",
        "feature_columns": list(X.columns),
        "training_rows": int(len(frame)),
        "model_parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        },
        "validation_metrics_original_price_scale": metrics,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "training_rows": len(frame),
        "validation_rows": len(X_valid),
        "model_path": str(model_path),
        "feature_count": X.shape[1],
    }


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Train it first with `python main.py train`."
        )
    # Only load model artifacts produced by this project or otherwise trusted.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Model artifact has an unsupported format. Retrain using this project's code.")
    return bundle


def prepare_prediction_features(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Prediction input must be a non-empty pandas DataFrame.")
    features = frame.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()
    expected = list(bundle["feature_columns"])
    missing = [column for column in expected if column not in features.columns]
    extra = [column for column in features.columns if column not in expected]
    if missing:
        raise ValueError(f"Prediction CSV is missing required feature columns: {missing[:10]}")
    # Preserve training-time feature order and ignore unrelated columns.
    return features.loc[:, expected]


def predict_frame(frame: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    features = prepare_prediction_features(frame, bundle)
    return predict_original_scale(bundle["model"], features)


def create_submission(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path,
) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {input_path}")
    input_frame = pd.read_csv(input_path)
    if input_frame.empty:
        raise ValueError("Prediction CSV contains no rows.")
    bundle = load_model_bundle(model_path)
    predictions = predict_frame(input_frame, bundle)
    ids = input_frame[ID_COLUMN] if ID_COLUMN in input_frame.columns else pd.Series(
        np.arange(1, len(input_frame) + 1), name=ID_COLUMN
    )
    submission = pd.DataFrame({ID_COLUMN: ids, TARGET: predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
