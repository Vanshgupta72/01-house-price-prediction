import argparse
from pathlib import Path

from src.car_price import predict_csv

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Predict Ford used-car prices from a CSV.")
    parser.add_argument("--input", type=Path, required=True, help="CSV containing the same feature columns as training data.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "ford_car_price.joblib")
    args = parser.parse_args()
    result = predict_csv(args.input, args.output, args.model)
    print(f"Generated {len(result)} predictions: {args.output}")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
