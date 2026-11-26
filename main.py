from __future__ import annotations

import argparse
from pathlib import Path

from src.house_price import train_model

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Ames house-price XGBoost model."
    )
    subparsers = parser.add_subparsers(dest="command", required=False)
    train = subparsers.add_parser("train", help="Train, evaluate, and save the model.")
    train.add_argument("--data", type=Path, default=ROOT / "data" / "train.csv")
    train.add_argument("--model-out", type=Path, default=ROOT / "artifacts" / "house_price_model.joblib")
    train.add_argument("--metrics-out", type=Path, default=ROOT / "reports" / "validation_metrics.json")
    train.add_argument("--estimators", type=int, default=500)
    train.add_argument("--max-depth", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=0.03)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--validation-size", type=float, default=0.2)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command in (None, "train"):
        result = train_model(
            data_path=getattr(args, "data", ROOT / "data" / "train.csv"),
            model_path=getattr(args, "model_out", ROOT / "artifacts" / "house_price_model.joblib"),
            metrics_path=getattr(args, "metrics_out", ROOT / "reports" / "validation_metrics.json"),
            n_estimators=getattr(args, "estimators", 500),
            max_depth=getattr(args, "max_depth", 3),
            learning_rate=getattr(args, "learning_rate", 0.03),
            random_state=getattr(args, "random_state", 42),
            validation_size=getattr(args, "validation_size", 0.2),
        )
        print("\nTraining completed.")
        print(f"Training rows : {result['training_rows']}")
        print(f"Validation rows: {result['validation_rows']}")
        print(f"Features      : {result['feature_count']}")
        print(f"MAE           : ${result['metrics']['mae']:,.2f}")
        print(f"RMSE          : ${result['metrics']['rmse']:,.2f}")
        print(f"R²            : {result['metrics']['r2']:.4f}")
        print(f"Model saved   : {result['model_path']}")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
