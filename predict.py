from __future__ import annotations

import argparse
from pathlib import Path

from src.house_price import create_submission

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Predict house prices from a CSV file.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "house_price_model.joblib")
    args = parser.parse_args()

    result = create_submission(args.input, args.output, args.model)
    print(f"Generated {len(result)} predictions.")
    print(f"Saved submission to: {args.output}")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
