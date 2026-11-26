import argparse
from pathlib import Path

from src.car_price import train_model

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train the Ford used-car price model.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "ford.csv")
    parser.add_argument("--model-out", type=Path, default=ROOT / "artifacts" / "ford_car_price.joblib")
    parser.add_argument("--metrics-out", type=Path, default=ROOT / "reports" / "validation_metrics.json")
    parser.add_argument("--estimators", type=int, default=350)
    args = parser.parse_args()
    metrics = train_model(args.data, args.model_out, args.metrics_out, n_estimators=args.estimators)
    print("Validation metrics (original price scale):")
    for key, value in metrics.items():
        print(f"{key.upper()}: {value:,.4f}")
    print(f"Saved model: {args.model_out}")


if __name__ == "__main__":
    main()
