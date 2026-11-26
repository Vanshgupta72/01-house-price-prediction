from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "house_price_xgboost.pkl"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"


def build_default_row(training_data: pd.DataFrame, feature_names) -> pd.DataFrame:
    """Build a one-row input using training-set medians/modes as defaults."""
    defaults = {}
    for column in feature_names:
        if column not in training_data.columns:
            raise ValueError(f"Required model feature '{column}' is missing from train.csv")

        series = training_data[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = series.median()
        else:
            modes = series.mode(dropna=True)
            defaults[column] = modes.iloc[0] if not modes.empty else "Unknown"

    return pd.DataFrame([defaults], columns=list(feature_names))


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run `python main.py` to train it first."
        )
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    model = joblib.load(MODEL_PATH)
    training_data = pd.read_csv(TRAIN_PATH)
    feature_names = list(getattr(model, "feature_names_in_", []))

    if not feature_names:
        raise RuntimeError(
            "The saved model does not expose feature_names_in_. "
            "Please retrain it with the included main.py."
        )

    sample = build_default_row(training_data, feature_names)

    prompts = {
        "OverallQual": ("Overall quality (1-10)", int),
        "GrLivArea": ("Above-ground living area (sq ft)", float),
        "YearBuilt": ("Year built", int),
        "TotalBsmtSF": ("Total basement area (sq ft)", float),
        "GarageCars": ("Garage capacity (cars)", float),
        "GarageArea": ("Garage area (sq ft)", float),
        "BedroomAbvGr": ("Bedrooms above ground", int),
        "FullBath": ("Full bathrooms", int),
        "HalfBath": ("Half bathrooms", int),
    }

    print("Enter key property details. Other features use training-data median/mode defaults.")
    for column, (label, converter) in prompts.items():
        if column in sample.columns:
            raw_value = input(f"{label}: ").strip()
            if raw_value:
                sample.loc[0, column] = converter(raw_value)

    prediction = float(model.predict(sample)[0])
    print(f"\nEstimated sale price: ${prediction:,.2f}")


if __name__ == "__main__":
    main()
