# Ford Car Price Prediction

A completed machine-learning workflow for predicting used Ford car prices.

## Dataset required

The original archive did **not** include `ford.csv`. Place the dataset at:

```text
data/ford.csv
```

The CSV must contain a numeric `price` target column. All remaining columns are treated as model features. The common Ford dataset fields include model, year, transmission, mileage, fuelType, tax, mpg, and engineSize.

Do not publish the dataset until you have verified its source and redistribution license.

## Features

- Numeric median imputation and categorical imputation.
- One-hot encoding with unseen-category handling.
- Random Forest regression.
- `log1p` target transformation with inverse transformation at prediction time.
- Holdout validation with MAE, RMSE, and R².
- Model serialization and batch CSV predictions.
- Reusable pipeline code and command-line interface.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train.py
```

## Predict

```powershell
python predict.py --input data/new_cars.csv
```

The prediction CSV must have the same feature columns as the training dataset. An optional `id` column is copied into the output.

## Tests

The root project test suite includes synthetic-data tests for this project. Actual Ford dataset training remains blocked until `data/ford.csv` is supplied.
