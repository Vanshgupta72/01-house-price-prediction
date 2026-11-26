# House Price Prediction — XGBoost

A reproducible regression project using an end-to-end scikit-learn preprocessing pipeline and XGBoost. It includes a separate Ford used-car price prediction project.

## Main project: Ames House Price Prediction

### What was fixed and improved

- Replaced fragile, manually hard-coded missing-value handling with fitted preprocessing pipelines.
- Kept imputation and one-hot encoding inside the estimator pipeline to prevent preprocessing leakage.
- Removed the non-predictive `Id` column from model features.
- Applied `log1p` to the target during training and correctly inverted predictions with `expm1`.
- Evaluated the model on a held-out validation split in the original price scale.
- Refit the final model on all labeled rows after validation.
- Added reproducible model configuration, model metadata, input-schema validation, CSV prediction, and automated tests.
- Preserved original source files and the supplied pickle under `legacy/`. The legacy pickle is not loaded by the new application.

### Requirements

- Python 3.10+
- See `requirements.txt`.

### Windows setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Train and evaluate

```powershell
python main.py train
```

The command:
1. Loads `data/train.csv`.
2. Creates a reproducible 80/20 holdout split.
3. Trains and evaluates XGBoost.
4. Reports MAE, RMSE, and R² in the original SalePrice units.
5. Refits on the complete training set.
6. Saves the model to `artifacts/house_price_model.joblib` and validation metrics to `reports/validation_metrics.json`.

You can change training settings:

```powershell
python main.py train --estimators 700 --max-depth 3 --learning-rate 0.03
```

### Generate predictions / Kaggle-style submission

After training:

```powershell
python predict.py
```

Default input: `data/test.csv`  
Default output: `outputs/submission.csv`

For a custom input:

```powershell
python predict.py --input path\to\houses.csv --output outputs\custom_predictions.csv
```

The prediction input must contain the same feature columns as the training CSV. Extra columns are ignored; missing required feature columns produce a clear error.

### Run tests

```powershell
python -m unittest discover -s tests -v
```

### Repository layout

```text
.
├── data/                       # supplied train/test CSVs and data notes
├── src/                        # reusable training/prediction logic
├── tests/                      # automated unit/integration tests
├── artifacts/                  # generated model (ignored by Git)
├── reports/                    # generated validation metrics (ignored by Git)
├── outputs/                    # generated predictions (ignored by Git)
├── 01_car_price_prediction/    # separate Ford used-car project
├── legacy/                     # original scripts and untrusted old model
├── main.py
├── predict.py
├── requirements.txt
└── README.md
```

## Separate project: Ford Car Price Prediction

See [`01_car_price_prediction/README.md`](01_car_price_prediction/README.md).

The uploaded archive did not contain `ford.csv`. The Ford project is now a complete, runnable training/prediction application, but actual dataset-based training cannot be completed until you place the correctly licensed Ford dataset at `01_car_price_prediction/data/ford.csv`.

## Security and publishing notes

- Only load model files that you trained yourself or otherwise trust. Pickle/joblib files can execute code when loaded.
- Generated model files, predictions, virtual environments, and local datasets are excluded by `.gitignore` where appropriate.
- Verify the source and redistribution license of the supplied datasets before committing them to a public repository.
- No license file is invented. Choose a repository license only if you have the right to license all included original code and data.
